import os
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import numpy as np
import pandas as pd
import shutil
import traceback

from freezingeffect.selection_of_ROIs import analyze_and_get_histograms, load_data_mm, generate_pixel_image, save_parameters

def collect_data_propagated(WM, path_align_folder, propagation_list, output_folders):
    """
    check_outliers is function checking if a ROI should be removed because if it is an outlier 
    (defined as ROI moving from grey/white matter to white/grey matter or background)

    Parameters
    ----------
    WM : boolean
        indicates if we are working with grey or white matter
    new_folder_names : list
        the folders of the measurements after formalin fixation
    new_dates : list
        the dates of the measurements after formalin fixation
    old_folder_name : str
        the name of the folder of the measurements before formalin fixation
    old_date : str
        the date of the measurements before formalin fixation
    path_folder : str
        the path to the measurement made before formalin fixation
    propagation_list : dict
        a dicationnary linking the information of the alignment and the folder currently analyzed
    output_folders : dict
    
    Returns
    ----------
    corrupt : boolean
        indicates if the ROI is an outlier
    """
    if WM:
        type_ = 'WM'
    else:
        type_ = 'GM'

    mask_matter_afters = []
    mask_matter_after_opposites = []
    
    
    for folder in propagation_list[0][1][1:]:
        path_fol = os.path.join(propagation_list[0][2], folder)
        # _ = get_masks(path_fol, bg = False)
        path_annotation = os.path.join(path_fol, 'annotation')
        WM_mask = plt.imread(os.path.join(path_annotation, 'WM_merged.png'))
        GM_mask = plt.imread(os.path.join(path_annotation, 'GM_merged.png'))
        if WM:
            mask_matter_afters.append(WM_mask)
            mask_matter_after_opposites.append(GM_mask)
        else:
            mask_matter_afters.append(GM_mask)
            mask_matter_after_opposites.append(WM_mask)
        
    remove = []
    for element in propagation_list:
        new_folder_name, all_folders, path_folders, wavelength, path_alignment, square_size = element
        for _, (all_folder, mask_matter_after, mask_matter_after_opposite) in enumerate(zip(all_folders[1:], 
                                                                    mask_matter_afters, mask_matter_after_opposites)):

            to_remove = check_outliers_propagation([all_folder], path_alignment, new_folder_name, mask_matter_after, 
                                                       mask_matter_after_opposite, path_align_folder, elastic = True)

            if len(to_remove) > 0:
                remove.append([to_remove[0], all_folder])
                
            data = propagate_measurements(new_folder_name, [all_folder], path_folders, wavelength, output_folders[path_alignment],  square_size, 
                                          mask_matter_after, mask_matter_after_opposite, path_align_folder, check_outliers_bool = False, create_dir = True)

            with open(os.path.join(path_folders, all_folder, 'polarimetry', '550nm', '50x50_images', 
                                new_folder_name + '_align', 'data_raw' + '.pickle'), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(path_folders, all_folders[0], 'polarimetry', '550nm', '50x50_images', 
                        'to_rm_' + type_ + '.pickle'), 'wb') as handle:
        pickle.dump(remove, handle, protocol=pickle.HIGHEST_PROTOCOL)

    generate_summary_file(propagation_list)
    

def check_outliers_propagation(all_folders, path_alignment, new_folder_name, mask_matter_after, mask_matter_after_opposite, path_align_folder,
                               elastic = True):
    """
    check_outliers_propagation is the master function to select and load the mask associated to a specific ROI, and to verify
    if it is an outlier (defined as ROI moving from grey/white matter to white/grey matter or background)

    Parameters
    ----------
    all_folders : list
        the names of all folders
    path_alignment : str
        the path to the alignement folder
    new_folder_name : str
        the 50x50 folder
    mask_matter_after, mask_matter_after_opposite : array of shape (516, 388)
        the annotation masks (one for the same tissue type and one for the opposite)
    elastic : bool
        indicates if we are using elastic registration (default : True)
    check_outliers_bool : bool
        indicates if we the outliers should be checked (default : True)
    
    Returns
    ----------
    to_remove : list
        the list of the ROIs to remove for further analyses
    """    
    path_align_folder = path_align_folder.replace('to_align', 'aligned')
    elastic = True

    for directory in os.listdir(path_align_folder):
        if path_alignment.split('\\')[-1] in directory:
            path_aligned = os.path.join(path_align_folder, directory)
    assert path_aligned != None

    to_remove = []
    
    _ = path_alignment.split('/')[-1].split('__')[0]
    
    for folder in all_folders:

        path_image = None
        img_of_interest = []
        for img in os.listdir(os.path.join(path_aligned, 'invReg')):
            if folder in img and '_PrpgTo_' in img and 'AffineElastic_TransformParameters' in img and '.png' in img:
                img_of_interest.append(img)
        assert len(img_of_interest) == 2

        for img in img_of_interest:
            if img.endswith('1.png') and elastic:
                path_image = path_aligned + '/invReg/' + img
            elif img.endswith('0.png') and not elastic:
                path_image = path_aligned + '/invReg/' + img
            
        mask = load_propagated_mask(path_image, new_folder_name)
        
        if check_outliers(mask, mask_matter_after, mask_matter_after_opposite):
            to_remove.append(new_folder_name)
        
    return to_remove


def load_propagated_mask(path_image, new_folder_name = 'WM_1'):
    """
    load the propagated mask from the image loacated at the path given as an input

    Parameters
    ----------
    path_image : str
        the path to the image
    new_folder_name : str
        the 50x50 folder (default : 'WM_1')
    
    Returns
    ----------
    imnp : array of shape (516, 388)
        the mask for the ROI
    """    
    val = int(new_folder_name.split('_')[-1])
    
    im = Image.open(path_image)
    imnp = np.array(im)
    
    for idx, x in enumerate(imnp):
        for idy, y in enumerate(x):
            if y != val :
                imnp[idx, idy] = 0
    return imnp


def check_outliers(mask, mask_matter_after, mask_matter_after_opposite):
    """
    check_outliers is function checking if a ROI should be removed because if it is an outlier 
    (defined as ROI moving from grey/white matter to white/grey matter or background)

    Parameters
    ----------
    mask : array of shape (516, 388)
        the mask indicating the location of the ROI
    mask_matter_after, mask_matter_after_opposite : array of shape (516, 388)
        the annotation masks (one for the same tissue type and one for the opposite)
    
    Returns
    ----------
    corrupt : boolean
        indicates if the ROI is an outlier
    """
    corrupt = False
    
    opposite = 0
    same = 0
    total = 0
    
    for idx, x in enumerate(mask):
        for idy, y in enumerate(x):
            if y > 0:
                total += 1
                
                # 1. if the ROI is located at the border
                if idx == 0 or idy == 0:
                    corrupt = True
                elif idx == mask.shape[0] - 1 or idx == mask.shape[1] - 1:
                    corrupt = True
                
                # 2. if the pixel is labelled with the same tissue type 
                elif mask_matter_after[idx, idy] == 1:
                    same += 1
                
                # 3. if the pixel is labelled with another tissue type 
                elif mask_matter_after_opposite[idx, idy] == 1:
                    opposite += 1
    
    try:
        # check if at least 50% of the pixels are the same tissue type, or at least 30% and the rest is located in background
        if same/total > 0.50 or (same/total > 0.30 and same/opposite == 0):
            pass
        else:
            corrupt = True
    except:
        pass

    return corrupt


def propagate_measurements(new_folder_name, all_folders, path_folders, wavelength, path_alignment, square_size, mask_matter_after, 
                           mask_matter_after_opposite, path_align_folder, check_outliers_bool = False, create_dir = False):
    """
    propagate_measurements is the master function used to propagate the ROIs and collect the data for the 
    measurement made after FF for one ROI

    Parameters
    ----------
    new_folder_name : str
        the name of the 50x50 folder for a specific ROI 
    all_folders : list
        the folders of the measurements after formalin fixation
    path_folders : str
        the path to the folder containing all the measurements
    wavelength : int
        the wavelenght currently analysed
    path_alignment : str
        the path to the aligned folder
    square_size : int
        the size of the ROI squares
    mask_matter_after, mask_matter_after_opposite : array of shape (516, 388)
        the annotation masks (one for the same tissue type and one for the opposite)
    create_dir : bool
        indicates if a new output dir should be created (default : False)
    check_outliers_bool : bool
        indicates if we the outliers should be checked (default : False)
    
    Returns
    ----------
    data : list
        the values of the polarimetric parameters for each folder studied
    dfs : list of pandas dataframe
        the statistic descriptors of the polarimetric parameters in a dataframe format
    aligned_images_path : str
        the path to the aligned images
    """
    output_directory, _ = get_output_directory_name(new_folder_name, all_folders, path_folders, wavelength)
    
    if create_dir:
        create_output_dir(output_directory, all_folders, path_folders, wavelength)
    
    data, dfs, aligned_images_path = get_data_propagation(output_directory, all_folders, path_folders, wavelength, path_alignment, square_size, 
                    new_folder_name, path_align_folder, create_dir = create_dir)

    return data, dfs, aligned_images_path
    
    
def get_output_directory_name(new_folder_name, all_folders, path_folders, wavelength):
    """
    get_output_directory_name allows to obtain the name of the 50x50 folder that should be used to store the data for the next ROI

    Parameters
    ----------
    new_folder_name : str
        the name of the previous 50x50 folder
    all_folders : list
        the folders of the measurements after formalin fixation
    path_folders : str
        the path to the folder containing all the measurements
    wavelength : int
        the wavelenght currently analysed
    
    Returns
    ----------
    new_folder_name : str 
        the name of the 50x50 folder that should be used to store the data for the next ROI
    """
    new_folder_name = new_folder_name + '_align'
    new_folder_free = True
    if not new_folder_name:
        new_folder_free = False
        
    number = 0
    for folder in all_folders:
        path_folder_50x50 = os.path.join(path_folders, folder, 'polarimetry', wavelength, '50x50_images')
        try:
            os.mkdir(path_folder_50x50)
        except:
            pass
        folders_ = os.listdir(path_folder_50x50)
        if new_folder_name in folders_:
            new_folder_free = False
        folder_number = len(folders_)
        if folder_number > number:
            number = folder_number
            
    if new_folder_free:
        return new_folder_name, path_folder_50x50
    else:
        return str(number) + '_align', path_folder_50x50
    
    
def create_output_dir(output_directory, all_folders, path_folders, wavelength):
    """
    create_output_dir creates the directory using the name given by get_output_directory_name

    Parameters
    ----------
    output_directory : str
        the name of the 50x50 folder that should be used to store the data for the next ROI
    all_folders : list
        the folders of the measurements after formalin fixation
    path_folders : str
        the path to the folder containing all the measurements
    wavelength : int
        the wavelenght currently analysed
        
    """
    for folder in all_folders:
        path_folder_50x50 = os.path.join(path_folders, folder, 'polarimetry', wavelength, '50x50_images')
        os.mkdir(os.path.join(path_folder_50x50, output_directory))
        
        
def get_data_propagation(output_directory, all_folders, path_folders, wavelength, path_alignment, square_size, 
                         new_folder_name, path_align_folder, elastic = True, create_dir = False):
    """
    this function allows to extract the values of the polarimetric parameters in the propagated ROIs

    Parameters
    ----------
    output_directory : str
        the name of the 50x50 folder in which the data should be stored
    all_folders : list
        the folders of the measurements after formalin fixation
    path_folders : str
        the path to the folder containing all the measurements
    wavelength : int
        the wavelenght currently analysed
    path_alignment : str
        the path to the aligned folder
    square_size : int
        the size of the ROI squares
    mask_matter_after, mask_matter_after_opposite : array of shape (516, 388)
        the annotation masks (one for the same tissue type and one for the opposite)
    create_dir : bool
        indicates if a new output dir should be created (default : False)
    check_outliers_bool : bool
        indicates if we the outliers should be checked (default : False)
    elastic : bool
        indicates if elastic registration was used (default : True)
    
    Returns
    ----------
    data : list
        the values of the polarimetric parameters for each folder studied
    dfs : list of pandas dataframe
        the statistic descriptors of the polarimetric parameters in a dataframe format
    aligned_images_path : str
        the path to the aligned images
    """
    # get the path to the folder containing the aligned images
    path_align_folder = path_align_folder.replace('to_align', 'aligned')
    path_aligned = None
    elastic = True
    for directory in os.listdir(path_align_folder):
        if path_alignment.split('\\')[-1] in directory:
            path_aligned = os.path.join(path_align_folder, directory)
    assert path_aligned != None
    try:
        aligned_images_path = os.path.join(path_aligned, 'aligned_images')
        os.mkdir(aligned_images_path)
    except FileExistsError:
        pass

    data = []
    dfs = []
    
    base_folder = path_alignment.split('\\')[-1].split('__')[0]
    for folder in all_folders:

        path_folder = os.path.join(path_folders, folder)
        path_output = os.path.join(path_folders, folder, 'polarimetry', str(wavelength), '50x50_images', output_directory)
        path_original_image = os.path.join(path_folders, folder, 'polarimetry', str(wavelength), path_folder.split('\\')[-1] + '_' + str(wavelength) + '_realsize.png')
        
        if folder == base_folder:
            raise ValueError
            #TODO: check what is happening here
            dfs.append([pd.read_csv(os.path.join(path_folder, 'polarimetry', str(wavelength) + 'nm', '50x50_images',
                               new_folder_name, '0_summary.csv')).set_index('parameter'), base_folder])
            selected_image = path_output.split('_align')[0] + '/selection.png'
            output_path = path_aligned_root + '/aligned_images/selection' + folder + '.png'
            shutil.copy(selected_image, output_path)
            
        else:
            
            # get the paths to the images
            path_image = None
            if folder in path_alignment:
                assert len(os.listdir(path_aligned + '/mask')) == 1
                path_image = path_aligned + '/mask/' + os.listdir(path_aligned + '/mask')[0]
            else:
                img_of_interest = []
                for img in os.listdir(path_aligned + '/invReg'):
                    if folder in img and '_PrpgTo_' in img and 'AffineElastic_TransformParameters' in img and '.png' in img:
                        img_of_interest.append(img)
                assert len(img_of_interest) == 2

                for img in img_of_interest:
                    if img.endswith('1.png') and elastic:
                        path_image = path_aligned + '/invReg/' + img
                    elif img.endswith('0.png') and not elastic:
                        path_image = path_aligned + '/invReg/' + img

            # load the propagated mask image
            mask = load_propagated_mask(path_image, new_folder_name)
            
            # load the data (polarimetric parameters)
            MM_maps, MM = load_data_mm(path_folder)
            data.append(analyze_and_get_histograms(MM_maps, MM, None, imnp = mask, )[1])
            _ = save_parameters(data[-1], path_output)

            if create_dir:
                # fig = generate_histogram(retardance, diattenua, azi, depol, path_output + '/', None)
                dfs.append([save_parameters(data[-1], path_output), folder])

            if create_dir:
                try:
                    _ = generate_pixel_image(None, path_original_image, path_output, mask = mask, 
                                                     path_save_align = os.path.join(path_aligned, 'aligned_images'), save = False)
                except:
                    pass

            # if create_dir:
                # selected_image = os.path.join(path_output, 'selection.png')
                # output_path = os.path.join(path_aligned, 'aligned_images', 'selection', folder + '.png')
                # print(output_path)
                # 0/0
                # shutil.copy(selected_image, output_path)
        
    return data, dfs, aligned_images_path


def generate_summary_file(propagation_list):
    """
    function that generates the summary files (.csv and .xlsx) and store them in the correct folders

    Parameters
    ----------
    propagation_list : dict
        a dicationnary linking the information of the alignment and the folder currently analyzed
    """ 
    new_folder_names = []
    all_folders = None
    path_folders = None
    wavelength = None
    
    for element in propagation_list:
        new_folder_name, all_folders, path_folders, wavelength, _, _ = element
        new_folder_names.append(new_folder_name)
        
    for new_folder_name in new_folder_names:
        
        # create the individual summaries, for each folder
        summaries = []
        for folder in all_folders:
            path_50x50_folder = get_path_50x50_folder(path_folders, folder, new_folder_name, wavelength)
            df = pd.read_csv(os.path.join(path_50x50_folder,  'summary.csv'))
            df['fname'] = [folder] * len(df)
            summaries.append(df)
            
        result = pd.concat(summaries)
        result = result.sort_values(['fname', 'parameter']).set_index(['fname', 'parameter'])

        
        # create the summaries for all the measurements combined
        for folder in all_folders:
            path_50x50_folder = get_path_50x50_folder(path_folders, folder, new_folder_name, wavelength, aligned = True)
            result.to_csv(os.path.join(path_50x50_folder, 'summary_aligned.csv'))
            result.to_excel(os.path.join(path_50x50_folder, 'summary_aligned.xlsx'))
            
            
def get_path_50x50_folder(path_folders, folder, new_folder_name, wavelength, aligned = False):
    """
    returns the path to the 50x50_folder in which the summary files (.csv and .xlsx) should be stored

    Parameters
    ----------
    path_folders : str
        the path to the data folder
    folder : str
        the name of the folder being considered
    new_folder_name : str
        the 50x50 folder
    wavelength : int
        the wavelenght currently studied
    aligned : bool
        indicated if we are using aligned folders (the target is different)
    
    Returns
    ----------
    path_50x50_folder : str
        the path to the 50x50_folder in which the summary files (.csv and .xlsx) should be stored
    """    
    path_50x50 = os.path.join(path_folders, folder, 'polarimetry', str(wavelength), '50x50_images')

    if os.path.exists(os.path.join(path_50x50, new_folder_name)):
        path_50x50_folder = os.path.join(path_50x50, new_folder_name)
    else:
        path_50x50_folder = os.path.join(path_50x50, new_folder_name + '_align')
    assert os.path.exists(path_50x50_folder)
    return path_50x50_folder