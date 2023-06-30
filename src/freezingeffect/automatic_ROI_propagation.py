from datetime import datetime
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matlab.engine

from freezingeffect.helpers import load_parameters_ROIs

def add_all_folders(path_folder, wavelength, path_alignment, pattern = '_FR_'):
    """
    add_all_folders is the function adding all the images from folders belonging to the same group of measurements (i.e. from the same sample) to be aligned
    
    Parameters
    ----------
    path_folder : str
        the path to the folder obtained before formalin fixation
    wavelength : int
        the wavelength currently studied
    pattern : str
        a pattern common to all the folders
    automatic : not used anymore
    
    Returns
    ----------
    path_alignment : str
        the path to the folder that will be aligned
    path_folders : str
        the path to the folder containing all the measurements subfolders
    all_folders : list of str
        the names of all the measurements subfolders corresponding to the same group of measurements (i.e. from the same sample)
    """
    all_folders = find_other_measurements(path_folder, pattern)
    
    folder_name = path_folder.split('\\')[-1]
    aligned = False
    path_aligned = None
    
    path_folders = '/'.join(path_folder.split('\\')[:-1])
    
    if aligned:
        path_alignment = path_aligned
    else:
        now = datetime.now()
        dt_string = path_folder.split('\\')[-1] +'__' + now.strftime("%d/%m/%Y %H:%M:%S").replace(' ', '_').replace('/', '_').replace(':', '_')
        path_alignment = os.path.join(path_alignment, dt_string)
        created = False
        while not created:
            try:
                os.mkdir(path_alignment)
                created = True
            except:
                print('Folder already exists, trying again...')
        os.mkdir(os.path.join(path_alignment, 'mask'))

        target = path_alignment
        for folder in all_folders:
            for file in os.listdir(os.path.join(path_folders, folder, 'polarimetry', wavelength)):
                if file.endswith('realsize.png'):
                    shutil.copy(os.path.join(path_folders, folder, 'polarimetry', wavelength, file), target)

        for file in os.listdir(path_alignment):
            if folder_name in file:
                old_name = path_alignment + '/' + file
                new_name = path_alignment + '/' + file.split('.')[0] + '_ref_align.png'
                os.rename(old_name, new_name)

    return path_alignment, path_folders, all_folders


def find_other_measurements(path_folder, pattern = '_FR_'):
    """
    find_other_measurements is a function allowing to find all the folders belonging to the same group of measurements (i.e. from the same sample)
    
    Parameters
    ----------
    path_folder : str
        the path to the folder obtained before formalin fixation
    pattern : str
        a pattern common to all the folders
    automatic : not used anymore
    end_pattern : str
        the pattern that needs to be matched at the end of the measurement folder
    
    Returns
    ----------
    img : image of shape (516, 388)
        the ROI overlaid onto the greyscale image
    """
    measurement_pattern = path_folder.split(pattern)[1]
    
    all_folders = []
    for folder in os.listdir('/'.join(path_folder.split('\\')[:-1])):
        if measurement_pattern in folder:
            all_folders.append(folder)

    return all_folders


def create_and_save_mask(imnp_mask):
    """
    create_and_save_mask overlays the ROI masks onto the greyscale image
    
    Parameters
    ----------
    imnp_mask : array of shape (516, 388)
        the ROI mask
    
    Returns
    ----------
    img : image of shape (516, 388)
        the ROI overlaid onto the greyscale image
    """
    imnp_masked = []
    for line in imnp_mask:
        line_mask = []
        for pixel in line:
            if pixel[-1] > 0:
                line_mask.append(0)
            else:
                line_mask.append(1)
        imnp_masked.append(line_mask)
        
    img = Image.new('1', (len(imnp_masked), len(imnp_masked[0])))
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = imnp_masked[i][j]
    img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
    return img


def generate_combined_mask(propagation_lists, path_align_folder):
    """
    generate_combined_mask generates a combined mask, compiling the masks generated for each ROI into a single one to fasten up the process
    
    Parameters
    ----------
    propagation_list : list
        a list containing the information about the ROIs (such as the origin measurement name, the square size...)
    
    Returns
    ----------
    propagation_list : list
        a list containing the information about the ROIs (such as the origin measurement name, the square size...) updated with the new path for the propagation folder
    """
    param_ROIs = load_parameters_ROIs()
    
    imgs = {}
    
    # obtain all the ROIs masks
    for file in os.listdir(path_align_folder):
        for img_path in os.listdir(os.path.join(path_align_folder, file, 'mask')):
            im = plt.imread(os.path.join(path_align_folder, file, 'mask', img_path))
            if 'WM' in img_path:
                imgs[int(img_path.split('WM_')[-1].split('_')[0])] = im
            elif 'GM' in img_path:
                imgs[int(img_path.split('GM_')[-1].split('_')[0]) + param_ROIs["max_number_of_random_squares"]] = im
    
    # compile them in a single mask
    base = np.zeros(imgs[1].shape)
    for val, img in imgs.items():
        assert val < 255
        for idx, x in enumerate(img):
            for idy, y in enumerate(x):
                if y != 0:
                    base[idx, idy] = val

    for img_path in os.listdir(os.path.join(path_align_folder, file, 'mask')):
        os.remove(os.path.join(path_align_folder, file, 'mask', img_path))

    mask = Image.fromarray(base)
    mask = mask.convert("L")
    mask.save(os.path.join(path_align_folder, file, 'mask', 'mask.png'))
    
    for folder in os.listdir(path_align_folder):
        path_folder = os.path.join(path_align_folder, folder)
        if file in folder:
            path_folder_propagation = path_folder
            for fname in os.listdir(path_folder):
                if 'P-T0_' in fname:
                    os.rename(os.path.join(path_folder, fname), 
                              os.path.join(path_folder, fname.replace('realsize', 'realsize_ref_align')))
        else:
            pass
            shutil.rmtree(path_folder)
            
            
    for _, propagation_list in propagation_lists.items():
        for prop in propagation_list:
            prop[-2] = path_folder_propagation
        
    return propagation_lists


def do_alignment(path_align_folder):
    """
    do_alignment is the function calling the matlab pipeline to align the images and propagate the ROIs
    
    Returns
    ----------
    output_folders : dict
        a dict linking the folder path to the 'to_align' to the ones in the 'aligned' subfolder
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'RegistrationElastix/temp/path_alignment_batch.txt'), 'w') as f:
        f.write(path_align_folder)
    f.close()

    FixPattern = '_ref_align'
    with open(os.path.join(dir_path, 'RegistrationElastix/temp/FixPattern.txt'), 'w') as f:
        f.write(FixPattern)
    f.close()

    Tag = 'AffineElastic'
    with open(os.path.join(dir_path, 'RegistrationElastix/temp/Tag.txt'), 'w') as f:
        f.write(Tag)
    f.close()

    generate_config_file(dir_path)

    eng = matlab.engine.start_matlab()
    path = os.path.join(dir_path, 'RegistrationElastix\RegistrationScripts')
    eng.cd(path, nargout=0)
    s = eng.genpath('0_NIfTI_IO')
    eng.addpath(s, nargout=0)
    eng.python_call(nargout=0)
    
    
def generate_config_file(dir_path):

    original_file = [r'% This is a configuration file with paths to external wrapped executables,',
    r'% or auxiliary configuration data employed in the compiled package.',
    r'% Please specify the local FilePaths - leave blank otherwise.',
    r'% All Entries follow the pattern: #ExecutableTAG \n strExecutableFILEPATH',
    '% Header lines starting with \'%\' will be treated as comments and ignored.',
    r'#EXE_Elastix',
    os.path.join(dir_path, r'elastix-5.0.1-win64\elastix.exe'),
    r'% Path HERE!',
    r'#EXE_Transformix',
    os.path.join(dir_path, r'elastix-5.0.1-win64\transformix.exe'),
    r'% Path HERE!',
    r'#LIB_SystemSharedLibs',
    r'# /lib/x86_64-linux-gnu/',
    '% Find the above path by running \'ldd <#EXE_Elastix>\' in the UNIX terminal',
    r'#LIB_Elastix',
    r'# /home/stefanohorao/Documents/software/elastix-501/Build/bin/',
    r'% Path HERE!']
    
    path = os.path.join(dir_path, 'RegistrationElastix', 'RegistrationScripts', 'configFilePaths.cfg')
    with open(path, 'w') as fp:
        for item in original_file:
            # write each item on a new line
            fp.write("%s\n" % item)
    fp.close()
    
    
def move_computed_folders(path_align_folder):
    """
    move_computed_folders is a function moving the aligned folders forom the subfolder 'to_align' to the subfolder 'aligned'
    
    Returns
    ----------
    output_folders : dict
        a dict linking the folder path to the 'to_align' to the ones in the 'aligned' subfolder
    """
    folder_names = []
    log_name = []
    for fname in os.listdir(path_align_folder):
        if not os.path.isfile(os.path.join(path_align_folder, fname)):
            folder_names.append(os.path.join(path_align_folder, fname))
        else:
            log_name.append(os.path.join(path_align_folder, fname))
    
    # create the dictionnary linking the folder path to the 'to_align' to the ones in the 'aligned' subfolder
    output_folders = {}
    for folder_name in folder_names:
        output_folder = path_align_folder.replace('to_align', 'aligned')
        if folder_name.split('\\')[-1] in os.listdir(output_folder):
            raise ValueError('The folder ' + folder_name.split('\\')[-1] + ' already exists in the folder ' + output_folder)
        else:
            output_folder = os.path.join(output_folder, folder_name.split('\\')[-1])
            output_folders[folder_name] = output_folder
      
    move_folders(output_folders, log_name, path_align_folder)
    return output_folders
    

def move_folders(output_folders, log_name, path_align_folder):
    for source, target in output_folders.items():
        shutil.move(source, '\\'.join(target.split('\\')[:-1]))
        
    # and the logboooks
    output_folder = os.path.join(path_align_folder.replace('to_align', 'aligned'), 'logbooks')
    try:
        os.mkdir(output_folder)
    except:
        pass
    for log in log_name:
        shutil.move(log, output_folder)
        
    return output_folders


def test_generate_combined_mask(propagation_lists, path_align_folder):
    """
    generate_combined_mask generates a combined mask, compiling the masks generated for each ROI into a single one to fasten up the process
    
    Parameters
    ----------
    propagation_list : list
        a list containing the information about the ROIs (such as the origin measurement name, the square size...)
    
    Returns
    ----------
    propagation_list : list
        a list containing the information about the ROIs (such as the origin measurement name, the square size...) updated with the new path for the propagation folder
    """
    param_ROIs = load_parameters_ROIs()
    
    imgs = {}
    
    # obtain all the ROIs masks
    for file in os.listdir(path_align_folder):
        for img_path in os.listdir(os.path.join(path_align_folder, file, 'mask')):
            im = plt.imread(os.path.join(path_align_folder, file, 'mask', img_path))
            if 'WM' in img_path:
                imgs[int(img_path.split('WM_')[-1].split('_')[0])] = im
            elif 'GM' in img_path:
                imgs[int(img_path.split('GM_')[-1].split('_')[0]) + param_ROIs["max_number_of_random_squares"]] = im
    
    # compile them in a single mask
    base = np.zeros(imgs[1].shape)
    for val, img in imgs.items():
        assert val < 255
        for idx, x in enumerate(img):
            for idy, y in enumerate(x):
                if y != 0:
                    base[idx, idy] = val

    for img_path in os.listdir(os.path.join(path_align_folder, file, 'mask')):
        os.remove(os.path.join(path_align_folder, file, 'mask', img_path))

    mask = Image.fromarray(base)
    mask = mask.convert("L")
    mask.save(os.path.join(path_align_folder, file, 'mask', 'mask.png'))
    
    for folder in os.listdir(path_align_folder):
        path_folder = os.path.join(path_align_folder, folder)
        if file in folder:
            path_folder_propagation = path_folder
            for fname in os.listdir(path_folder):
                if 'P-T0_' in fname:
                    os.rename(os.path.join(path_folder, fname), 
                              os.path.join(path_folder, fname.replace('realsize', 'realsize_ref_align')))
        else:
            pass
            shutil.rmtree(path_folder)
            
            
    for _, propagation_list in propagation_lists.items():
        for prop in propagation_list:
            prop[-2] = path_folder_propagation
        
    return propagation_lists