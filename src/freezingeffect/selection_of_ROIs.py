import os 
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import traceback
from scipy.stats import circmean, circstd
import pandas as pd

from freezingeffect.helpers import load_param_names_link, load_parameters_ROIs, load_histogram_parameters, load_parameter_maps
from freezingeffect import automatic_ROI_propagation 

def create_alignment_folder():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        os.mkdir(os.path.join(dir_path, 'alignment'))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(dir_path, 'alignment', 'to_align'))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(dir_path, 'temp'))
    except FileExistsError:
        pass
    return os.path.join(dir_path, 'temp'), os.path.join(dir_path, 'alignment', 'to_align')


def get_the_base_dirs(data_folder_path):
    base_dirs = []

    for folder in os.listdir(data_folder_path):
        if 'T0_' in folder:
            base_dirs.append(os.path.join(data_folder_path, folder))
            
        # 2. remove previously acquired data to allow the selection of new ROIs
        path_results = os.path.join(data_folder_path, folder, 'polarimetry/550nm/50x50_images')
        try:
            os.mkdir(path_results)
        except FileExistsError:
            pass
        
        for filename in os.listdir(path_results):
            file_path = os.path.join(path_results, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                
    return base_dirs
                

def clean_the_alignment_folders(path_align_folder):
    for fname in os.listdir(path_align_folder):
        file_path = os.path.join(path_align_folder, fname)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
            
def get_mask_matter_and_grid(path_folder, tissue_type):
    """
    get_mask_matter_and_grid allows to obtain the masks for white and grey matter for the folder analyzed and the grid to 
    register the ROIs as they are obtained

    Parameters
    ----------
    path_folder : str
        the path to the folder
    tissue_type : str
        the tissue type considered (i.e. 'WM' or 'GM')
        
    Returns
    -------
    mask_matter : array of shape(388,516)
        the annotation mask for white or grey matter
    grid : array of shape(388,516)
        the grid that will be used to register the ROIs as they are obtained
    """
    _ = get_masks(path_folder, bg = False)
    path_annotation = os.path.join(path_folder, 'annotation')
    if tissue_type == 'WM':
        WM_mask = plt.imread(os.path.join(path_annotation, 'WM_merged.png'))
        return WM_mask, np.zeros(WM_mask.shape)
    else:
        GM_mask = plt.imread(os.path.join(path_annotation, 'GM_merged.png'))
        return GM_mask, np.zeros(GM_mask.shape)


def get_masks(path, bg = True):
    """
    obtain the masks (white matter, grey matter and background) by combining previsously manually drawn masks

    Parameters
    ----------
    path : str
        the path to the folder containing the annotations considered

    Returns
    -------
    BG_merged : array of shape (388,516)
        the background mask
    GM_merged : array of shape (388,516)
        the grey matter mask
    WM_merged : array of shape (388,516)
        the white matter mask
    all_merged : array of shape (388,516)
        the three masks combined (one color = one class)
    """
    BG = []
    WM = []
    path = os.path.join(path, 'annotation')
    
    # add the masks to the different lists
    for file in os.listdir(path):
        if '.txt' in file:
            pass
        else:
            im = Image.open(os.path.join(path, file))
            imarray = np.array(im)
            
            if 'BG' in file and not 'merged' in file:
                BG.append(imarray)
            elif 'WM' in file and not 'merged' in file:
                WM.append(imarray)
            elif 'merged' in file or 'mask-viz' in file or 'mask_total' in file:
                pass
            else:
                raise(NotImplementedError)
    
    
    # combine the WM and BG masks
    WM = combine_masks(WM)
    BG = combine_masks(BG)
    GM = np.zeros(WM.shape)
    
    # return the merged masks
    return merge_masks(BG, WM, GM, path, bg)


def combine_masks(masks):
    """
    combine previsously manually drawn masks

    Parameters
    ----------
    masks : list of array of shape (388,516)
        the manually drawn masks

    Returns
    -------
    base : array of shape (388,516)
        the combined mask
    """
    if masks:
        # use the first mask as a base
        base = masks[0]
        
        # for each of the mask, search for pixels equal to 255 and add them as positive values to the base mask
        for id_, mask in enumerate(masks[1:]):
            for idx, x in enumerate(mask):
                for idy, y in enumerate(x):
                    if base[idx, idy] == 255 or y == 255:
                        base[idx, idy] = 255
    
    # if no mask is found, everything is set to 0
    else:
        base = np.zeros((388, 516))
    return base


def merge_masks(BG, WM, GM, path, bg):
    """
    merge masks is used to merge the previsously combined manually drawn masks

    Parameters
    ----------
    BG : array of shape (388,516)
        the combined background mask
    GM : array of shape (388,516)
        the combined grey matter mask
    WM : array of shape (388,516)
        the combined white matter mask
    path : str
        the path to the folder containing the annotations considered

    Returns
    -------
    BG_merged : array of shape (388,516)
        the merged background mask
    GM_merged : array of shape (388,516)
        the merged grey matter mask
    WM_merged : array of shape (388,516)
        the merged white matter mask
    all_merged : array of shape (388,516)
        the three masks combined (one color = one class)
    """
    WM_merged = np.zeros(WM.shape)
    BG_merged = np.zeros(WM.shape)
    GM_merged = np.zeros(WM.shape)
    all_merged = np.zeros(WM.shape)

    for idx, x in enumerate(WM):
        for idy, y in enumerate(x):
            
            if bg:
                # 1. check if the pixel is white matter
                if WM[idx, idy] == 255:
                    WM_merged[idx, idy] = 255
                    all_merged[idx, idy] = 255
                    
                # 2. if not, check if it is background
                elif BG[idx, idy] == 255:
                    BG_merged[idx, idy] = 255
                    all_merged[idx, idy] = 0
                
                # 3. if not, it is grey matter
                else:
                    GM_merged[idx, idy] = 255
                    all_merged[idx, idy] = 128
            
            else:
                # 1. check if it is background
                if WM[idx, idy] == 255:
                    WM_merged[idx, idy] = 255
                    all_merged[idx, idy] = 255

                # 2. if not, check if the pixel is white matter
                elif BG[idx, idy] == 255:
                    BG_merged[idx, idy] = 255
                    all_merged[idx, idy] = 0
                    
                # 3. if not, it is grey matter
                else:
                    GM_merged[idx, idy] = 255
                    all_merged[idx, idy] = 128

    # save the masks
    save_image(path, WM_merged, 'WM_merged')
    save_image(path, BG_merged, 'BG_merged')
    save_image(path, GM_merged, 'GM_merged')
    new_p = Image.fromarray(all_merged)
    if new_p.mode != 'L':
        new_p = new_p.convert('L')
        new_p.save(os.path.join(path, 'merged.jpeg'))
        new_p.save(os.path.join(path, 'merged.png'))
        
    return BG_merged, WM_merged, GM_merged, all_merged


def save_image(path, img, name):
    """
    save_image is used to save an image as a .png file

    Parameters
    ----------
    path : str
        the path to the folder containing the annotations considered
    img : array of shape (388,516)
        the image to be saved
    name : str
        the name of the image
    """
    new_p = Image.fromarray(img)
    if new_p.mode != 'L':
        new_p = new_p.convert('L')
        new_p.save(os.path.join(path, name + '.png'))
        
        
def create_the_masks(path_fixation_folder, match_sequence = 'FRE-FR'):
    """
    master function to create the combined match for all the folders located in path_fixation_folder

    Parameters
    ----------
    path_fixation_folder : str
        the path to the measurement folder
    match_sequence : str
        the sequence to match to be considered a measurement folder
    """
    folders = os.listdir(path_fixation_folder)

    folder_of_interests = []
    for folder in folders:
        if match_sequence in folder:
            folder_of_interests.append(os.path.join(path_fixation_folder, folder))

    for folder_of_interest in tqdm(folder_of_interests):
        _ = get_masks(folder_of_interest)
        
        
def load_data_mm(path_folder):
    """
    load_data_mm allows to load the Mueller Matrix and the parameters of interest (retardance, diattenuation, azimuth and depolarization)

    Parameters
    ----------
    path_folder : str
        the path to the folder
    wavelength : int or str
        the wavelength being considered (i.e. '550' or '650')
        
    Returns
    -------
    linear_retardance : array of shape(388,516)
    diattenuation : array of shape(388,516)
    azimuth : array of shape(388,516)
    depolarization : array of shape(388,516)
    mat : dict
        the Mueller matrix
    """
    param_ROIs = load_parameters_ROIs()
    mat = np.load(os.path.join(path_folder, 'polarimetry', param_ROIs['wavelength'], 'MM.npz'))
    param_names_link = load_param_names_link(inv = True)
    pol_parameters = {}
    for param_name, key_MM in param_names_link.items():
        pol_parameters[param_name] = load_and_verify_parameters(mat, key_MM)
        pass
    return pol_parameters, mat


def load_and_verify_parameters(mat, name):
    """
    select the array for the parameter and check the correct size

    Parameters
    ----------
    mat : dict
        the Mueller matrix
    name : str
        the key for the parameter of interest
        
    Returns
    -------
    out : array of shape(388,516)
        the values for the parameter of interest
    """
    out = mat[name]
    assert len(out) == 388
    assert len(out[0]) == 516
    return out


def square_selection(path_folder_temp, path_folder, WM, tissue_type, path_align_folder):
    param_ROIs = load_parameters_ROIs()
    wavelength = param_ROIs['wavelength']
    number_of_random_squares = param_ROIs['number_of_random_squares']
    path_folder_50x50 = os.path.join(path_folder, 'polarimetry', wavelength, '50x50_images')
    folder_name = path_folder.split('\\')[-1]
    
    MM_maps, MM = load_data_mm(path_folder)
    mask_matter, grid = get_mask_matter_and_grid(path_folder, tissue_type)
    
    continuation = True
    propagation_list = []
    valerr = False
    
    # counter of number of squares (for automatic mode)
    counter = 0
    
    # get the name of the annotated image
    if WM:
        path_img_annotated = 'img_annotated_WM.png'
    else:
        path_img_annotated = 'img_annotated_GM.png'
        
    path_image = os.path.join(path_folder, 'polarimetry', wavelength, folder_name + '_' + wavelength + '_realsize.png')
    
    if os.path.exists(os.path.join(path_folder_50x50, path_img_annotated)):
        pass
    else:
        shutil.copyfile(path_image, os.path.join(path_folder_50x50, path_img_annotated))
        
    while continuation and counter < number_of_random_squares and not valerr:
        
        # get the path of the image and of the new output folder
        path_output, new_folder_name = get_new_output(path_folder_50x50, WM)

        with open(os.path.join(path_folder_temp, 'image_path.txt'), 'w') as file:
            file.write(path_image)
            
        # if automatic mode, get automatic square using get_square_coordinates
        try:
            coordinates_long, grid = get_square_coordinates(mask_matter, param_ROIs['square_size'], grid, MM = MM)
            all_coordinates = [coordinates_long]
        except:
            traceback.print_exc()
            valerr = True     
        
        if not valerr:
            # write the coordinates to a txt file to reuse later
            textfile = open(os.path.join(path_output, 'coordinates.txt'), 'w')
            for element in coordinates_long:
                textfile.write(str(element) + "\n")
            textfile.close()

            # get the values for the square and generate the histograms
            try:
                imnp_mask_single, imnp_mask = histogram_analysis(all_coordinates, MM_maps, param_ROIs, coordinates_long, path_image, path_img_annotated, path_output, path_folder_50x50, 
                                                                 WM = WM, MM = MM)
            except:
                traceback.print_exc()
                imnp_mask = histogram_analysis(all_coordinates, MM_maps, param_ROIs, coordinates_long, path_image, path_img_annotated, path_output, path_folder_50x50, 
                                                                 WM = WM, MM = MM)

            try:
                im = Image.fromarray(imnp_mask_single.T)
                im.save(os.path.join(path_folder_50x50, path_img_annotated))

            except:
                traceback.print_exc()
                im = Image.fromarray(imnp_mask.T)
                im.save(os.path.join(path_folder_50x50, path_img_annotated))

            continuation = True
            propagation = True

            counter += 1

            # if yes, perform the propagation
            if propagation:
                path_alignment, path_folders, all_folders = automatic_ROI_propagation.add_all_folders(path_folder, wavelength, path_align_folder)
                img = automatic_ROI_propagation.create_and_save_mask(imnp_mask.T)
                new_folder_name = path_output.split('\\')[-1]
                img.save(os.path.join(path_alignment, 'mask', path_folder.split('\\')[-1] + '_' + new_folder_name + '_selection.png'))
                propagation_list.append([new_folder_name, all_folders, path_folders, wavelength, path_alignment, param_ROIs['square_size']])
                continuation = True
                propagation = True
            
    return propagation_list


def get_new_output(path_folder_50x50, WM):
    """
    get_new_output returns the new name for the folder in which the results should be outputted

    Parameters
    ----------
    path_folder_50x50 : str
        the path to the 50x50 folder in the folder for the measurement of interest
    WM : boolean  
        indicates if white or grey matter is analyzed
        
    Returns
    -------
    path_output : str
        the new name for the folder in which the results should be outputted
    """
    name = 'WM_' if WM else 'GM_'
    folders = os.listdir(path_folder_50x50)
    dirs = []
    for fol in folders:
        if os.path.isdir(os.path.join(path_folder_50x50, fol)) and fol.replace(name, '').isdecimal():
            dirs.append(fol)
    folder_number = len(dirs)
    path_output = os.path.join(path_folder_50x50, name + str(folder_number + 1))

    try:
        os.mkdir(path_output)
    except:
        raise ValueError('The folder already exists')
    
    folder_nb = 'WM_' + str(folder_number + 1) if WM else 'GM_' + str(folder_number + 1)
    return path_output, folder_nb


def get_square_coordinates(mask, square_size, grid, MM, coordinates = None):
    """
    get_square_coordinates allows to obtain a randomly selected ROI in the image, and to check for the fulfillement of:
        1. the presence of pixels labelled as the tissue type that is being analyzed 
        2. the presence of valid pixels in the ROI

    Parameters
    ----------
    mask : array of shape (388, 516)
        the mask for tissue type in the complete image
    square_size : int
        the size of the ROI square
    grid : array of shape (388, 516)
        the grid used to register the ROIs as they are obtained
    mat : dict
        the Mueller matrix
    coordinates : list of int    
        the minimum and maximum values for the ROI index
    
    Returns
    -------
    coordinates : list of int    
        the minimum and maximum values for the ROI index
    grid : array of shape (388, 516)
        the grid used to register the ROIs, updated for the new ROI
    """    
    found = False
    counter = 0
    
    while not found and counter < 1000:
        randomRow, randomCol = get_random_pixel(mask)
        if mask[randomRow, randomCol] == 0:
            counter += 1
        else:
            # print(select_region(mask.shape, mask, randomRow, randomCol))
            region, grided, coordinates = select_region(mask.shape, mask, randomRow, randomCol, square_size, grid)
            
            # check if the region is part of the correct tissue type...
            positive_mask = search_for_validity(region, 0, MM = MM, coordinates = coordinates)
            # ...and if it has not been selected before
            positive_grid = search_for_validity(grided, 1, MM = MM, coordinates = coordinates)
            
            positive = positive_mask and positive_grid
            if positive:
                found = True
                grid = update_grid(grid, coordinates)

            counter += 1
            
    if counter == 1000:
        pass
    else:
        return coordinates, grid


def get_random_pixel(mask):
    """
    get_random_pixel returns a randon column and row using a uniform distribution

    Parameters
    ----------
    mask : array of shape (388, 516)
        the mask for tissue type in the complete image

    
    Returns
    -------
    randomRow : int
        the row index
    randomCol : int
        the column index
    """
    randomRow = np.random.randint(mask.shape[0], size=1)
    randomCol = np.random.randint(mask.shape[1], size=1)
    return randomRow[0], randomCol[0]


def select_region(shape, mask, idx, idy, square_size, grid, border = 1.5, offset = 15):
    """
    select randomly a region in the image that is located at a distance > offset from the border of the image

    Parameters
    ----------
    shape : tuple
        the shape of the array
    mask : array of shape (388, 516)
        the mask for tissue type in the complete image
    idx, idy : int, int
        the index values of the pixel of interest
    grid : array of shape (388, 516)
        the grid used to register the ROIs as they are obtained
    border : double
        a scaling number for checking that the pixels in the region immediatly surrounding the ROI also belong to the same tissue type
    offset : double
        an offset ensuring that the ROIs are not located to close to the border of the image
    
    Returns
    -------
    mask : array
        the mask for tissue type in the ROI
    grid : array
        the grid 
    min_y, max_y, min_x, max_x : int
        the minimum and maximum values for the ROI index
    """
    max_x, min_x = None, None
    max_y, min_y = None, None
    
    # special cases - borders of the image
    if idx - border * (square_size // 2 + 1) - offset < 0:
        min_x = 0 + offset
        min_x_reg = 0 + offset - border * square_size // 2
        max_x = square_size + offset
        max_x_reg = border * square_size // 2 + offset
        
    if idy - border * (square_size // 2 + 1) - offset < 0:
        min_y = 0 + offset
        min_y_reg = 0 + offset - border * square_size  // 2
        max_y = square_size + offset
        max_y_reg = border * square_size // 2 + offset
        
    if idx + border * (square_size // 2 + 1) + offset > shape[0]:
        min_x = shape[0] - square_size - offset
        min_x_reg = shape[0] - offset - border * (square_size  // 2) 
        max_x = shape[0] - offset
        max_x_reg = shape[0] - offset + border * (square_size  // 2)
        
    if idy + border * (square_size // 2 + 1) + offset > shape[1]:
        min_y = shape[1] - square_size - offset
        min_y_reg = shape[1] - border * (square_size  // 2) - offset
        max_y = shape[1] - offset
        max_y_reg = shape[1] - offset + border * (square_size  // 2)
        
    # middle of the image
    if max_x == None and min_x == None:
        min_x = idx - (square_size//2)
        min_x_reg = idx - border * (square_size//2)
        max_x = idx + (square_size//2)
        max_x_reg = idx + border * (square_size//2)
        
    if max_y == None and min_y == None:
        min_y = idy - (square_size//2)
        min_y_reg = idy - border * (square_size//2)
        max_y = idy + (square_size//2)
        max_y_reg = idy + border * (square_size//2)

    return mask[int(min_x_reg): int(max_x_reg), int(min_y_reg):int(max_y_reg)], grid[int(min_x):int(max_x), 
                                                        int(min_y):int(max_y)], [min_y, max_y, min_x, max_x]
    
    
def search_for_validity(mask, idx, MM, coordinates = None):
    """
    search_for_validity allows to check if the ROI generated fullfills the following requirements:
        1. contains also pixels labelled as the tissue type studied
        2. contains more than 80% of valid pixels

    Parameters
    ----------
    mask : array 
        the mask for tissue type in the ROI
    idx : int
        the value against which to check the values in the mask (0 for the mask, 1 for the grid)
    mat : dict
        the Mueller matrix
    coordinates : list of int    
        the minimum and maximum values for the ROI index
        
    Returns
    -------
    postitive : boolean
        indicates if the consitions were fulfilled
    """
    positive = True
        
    # 1. contains also pixels labelled as the tissue type studied
    for row in mask:
        for y in row:
            if y == idx:
                positive = False
    
    # 2. contains more than 80% of valid pixels
    if positive and idx == 1:
        positive = sum(sum(MM['Msk'][coordinates[2]:coordinates[3], coordinates[0]:coordinates[1]])) > 0.6 * mask.shape[0]*mask.shape[1]
        if not positive:
            pass
    
    return positive


def update_grid(grided, coordinates):
    """
    update_grid updates the grid and add the newly generated ROI

    Parameters
    ----------
    grided : array of shape (388, 516) 
        the grid used to register the ROIs as they are obtained
    coordinates : list of int    
        the minimum and maximum values for the ROI index
        
    Returns
    -------
    grided : array of shape (388, 516) 
        the updated grid
    """
    for idx, x in enumerate(grided):
        for idy, y in enumerate(x):
            if coordinates[0] <= idy <= coordinates[1] and coordinates[2] <= idx <= coordinates[3]:
                grided[idx, idy] = 1
    return grided


def histogram_analysis(all_coordinates, MM_maps, param_ROIs, coordinates_long, path_image, path_img_annotated, path_output, path_folder_50x50, WM, MM):
    """
    histogram_analysis is the master function used to analyze the polarimetric parameters in the ROI, save the statistical descriptors and generate an image with the pixel in the new ROI highlighted

    Parameters
    ----------
    all_coordinates : list of list
        the coordinated of the squares that is being studied (here the length of the list is 0)
    linear_retardance : array of shape (388, 516)
    diattenuation : array of shape (388, 516)
    azimuth : array of shape (388, 516)
    depolarization : array of shape (388, 516)
    square_number : int
        the number of squares (here set to 1)
    square_size_horizontal, square_size_vertical : int, int
        the size of the square / rectangle in the horizontal and vertical axis (in this case, they are the same)
    square_size : int
        the size of the square
    orientation : str
        horizontal or vertical
    coordinates_long : not used anymore
    path_image, path_image_ori : str, str
        the path to the greyscale images on which to highlight the ROI
    path_output : str
        the path to the folder in which to save the image
    WM : boolean
        boolean indicating if white or grey matter is currently being analyzed
    mat : dict
        the mueller matrix
    
    Returns
    -------
    imnp_mask : array of shape (388, 516)
        the image with the new ROI highlighted
    """
    path_image_ori = os.path.join('/'.join(path_output.split('/')[:-1]), path_img_annotated)
    square_number = 1
    orientation = None
    square_size_horizontal, square_size_vertical = param_ROIs['square_size'], param_ROIs['square_size']
    
    data = []
    for _, coordinates in enumerate(all_coordinates):
        data.append(analyze_and_get_histograms(MM_maps, MM, coordinates)[1])
        _ = parameters_histograms(data[-1], path_output)
        _ = save_parameters(data[-1], path_output)
        
    imnp_mask = generate_pixel_image(coordinates_long, path_image, path_output, [square_number, [square_size_horizontal, square_size_vertical], orientation], WM = True)
    
    try:
        imnp_mask_single = generate_pixel_image(coordinates_long, os.path.join(path_folder_50x50, path_image_ori), path_output, 
                                [square_number, [square_size_horizontal, square_size_vertical], orientation], combined = True, WM = WM)
    except:
        traceback.print_exc()
        pass
    
    generate_summary_file_series(path_output, square_number, WM)
    
    try:
        return imnp_mask_single, imnp_mask
    except:
        return imnp_mask
    
    
def analyze_and_get_histograms(MM_maps, MM, coordinates, imnp = None, angle = 0):
    """
    analyze_and_get_histograms extracts the values of the parameters in the ROI, as well as the statistical descriptors

    Parameters
    ----------
    linear_retardance : array of shape (388, 516)
    diattenuation : array of shape (388, 516)
    azimuth : array of shape (388, 516)
    depolarization : array of shape (388, 516)
    mat : dict
        the mueller matrix
    imnp : array of shape (388, 516)  
        a mask representing the pixels for which values that should be extracted (optional)
    angle : int
        the angle by which the azimuth should be corrected
        
    Returns
    -------
    retardance, diattenua, azi, depol : lists
        the values of the parameters in the ROI, as well as the statistical descriptors 
    """
    parameters = load_histogram_parameters()
    
    if type(imnp) == np.ndarray:
        masked = True
    else:
        masked = False
        
    if masked:
        coordinates = imnp
    else:
        pass
    
    param_links = load_param_names_link(inv = True)
    results = {}
    for param, key in param_links.items():
        results[param] = get_area_of_interest(coordinates, MM_maps[param], parameters[param], masked, param, mat = MM)

    return coordinates, results


def get_area_of_interest(params, matrix, param, masked, parameter, mat):
    """
    get_area_of_interest extracts the values of one parameter in the ROI, as well as the statistical descriptors

    Parameters
    ----------
    params : 
    matrix : array of shape (388, 516)
        the polarimetric parameter values
    param : list
        the parameters to use to build the histogram to extract the 'max' descriptor
    masked : array of shape (388, 516)
        optional, mask that represent the pixels for which the values should be extracted
    angle : int
        the angle by which the azimuth should be corrected
    mat : dict
        the mueller matrix
        
    Returns
    -------
    mean, stdev, maximum, listed, median : list
        the values of one parameter in the ROI, as well as the statistical descriptors
    """
    if masked:
        listed = []
        for idx_x, x in enumerate(params):
            for idx_y, y in enumerate(x):
                if y != 0 and mat['Msk'][idx_x, idx_y]:
                    # listed.append((matrix[idx_x][idx_y] - angle)%180)
                    listed.append(matrix[idx_x][idx_y])
    else:
        [y_min, y_max, x_min, x_max] = params
        listed = []
        for idx_x, x in enumerate(mat['Msk']):
            for idx_y, y in enumerate(x):
                if x_min <= idx_x < x_max and y_min <= idx_y < y_max and y:
                    # listed.append((matrix[idx_x][idx_y] - angle)%180)
                    listed.append(matrix[idx_x][idx_y])

    if parameter == 'azimuth':
        mean = circmean(listed, high=180)
        stdev = circstd(listed, high=180)
        median = mean
    else:
        mean = np.mean(listed)
        stdev = np.std(listed)
        median = np.median(listed)

    bins = np.linspace(param['borders'][0], param['borders'][1], num = param['num_bins'])
    data = plt.hist(listed, bins = bins)
    plt.close()
    arr = data[0]
    max_idx = np.where(arr == np.amax(arr))[0][0]
    maximum = data[1][max_idx]
    
    return mean, stdev, maximum, median, listed



def parameters_histograms(data: dict, path_folder: str):
    """
    generate the histogram for the four parameters

    Parameters
    ----------
    MuellerMatrices : dict
        the dictionnary containing the computed Mueller Matrices
    folder : str
        the name of the current processed folder
    max_ : bool
        boolean indicating wether or not the max_ should be printed
    """
    parameters_map = load_parameter_maps()
    param_links = load_param_names_link()
    
    try:
        parameters_map.pop('M11')
    except:
        pass
    
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,11))
    
    for i, (key, param) in zip(range(0,4), parameters_map.items()):
        row = i%2
        col = i//2
        ax = axes[row, col]
        
        # change the range of the histograms
        if param[2]:
            range_hist = (0, 1)
        elif param[1]:
            range_hist = (0, 180)
        elif param[3]:
            range_hist = (0, 0.20)
        else:
            range_hist = (0, 60)
        
        y, x = np.histogram(
            data[param_links[key]][-1],
            bins=75,
            density=False,
            range = range_hist)
        
        x_plot = []
        for idx, _ in enumerate(x):
            try: 
                x_plot.append((x[idx] + x[idx + 1]) / 2)
            except:
                assert len(x_plot) == 75
        
        # get the mean, max and std
        max_ = x[np.argmax(y)]
        mean = np.nanmean(data[param_links[key]][-1])
        std = np.nanstd(data[param_links[key]][-1])
        
        y = y / np.max(y)
        
        # plot the histogram
        ax.plot(x_plot,y, c = 'black', linewidth=3)
        ax.axis(ymin=0,ymax=1.5)
        ax.locator_params(axis='y', nbins=4)
        ax.locator_params(axis='x', nbins=5)
    
        if max_:
            ax.text(0.75, 0.85, '$\mu$ = {:.3f}\n$\sigma$ = {:.3f}'.format(mean, std, max_), 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, 
                    fontsize=25, fontweight = 'bold')
        else:
            ax.text(0.75, 0.85, '$\mu$ = {:.3f}\n$\sigma$ = {:.3f}'.format(mean, std), 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, 
                    fontsize=25, fontweight = 'bold')
        
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(22)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(22)
            tick.label1.set_fontweight('bold')
            
        ax.set_title(param[0], fontdict = {'fontsize': 30, 'fontweight': 'bold'})
        ax.set_ylabel('Normalized pixel number', fontdict = {'fontsize': 25, 'fontweight': 'bold'})
        
    # save the figures
    plt.tight_layout()
    plt.savefig(os.path.join(path_folder, 'histogram.png'))
    plt.savefig(os.path.join(path_folder, 'histogram.pdf'))
    plt.close()
    
    
def save_parameters(data, path_output):
    """
    save_parameters retrieves the statistical descriptors for all the parameters and save a summary .xlsx and .csv files

    Parameters
    ----------
    retardance, diattenua, azi, depol : lists
        the values of the parameters in the ROI, as well as the statistical descriptors
    path_output : str
        the path in which the summary files should be saved
    square_size : int
        the size of the squares
    idx : not used
        
    Returns
    -------
    df : pandas dataframe
        the summary dataframe
    """
    parameters_names = load_parameter_maps()
    parameters_names.pop('M11')
    parameters_names = list(parameters_names.keys())
    param_links = load_param_names_link()
    
    params = []
    for param in parameters_names:
        params.append(get_params_summary(data[param_links[param]], param_links[param]))

    df = pd.DataFrame(params, columns = ['parameter', 'mean', 'stdev', 'max', 'median'])
    df = df.set_index('parameter')
    fn = []
    for i in range(len(df)):
        fn.append(str(None))
    df['square_size'] = fn
    df.to_csv(os.path.join(path_output, 'summary.csv'))
    df.to_excel(os.path.join(path_output, 'summary.xlsx'))
        
    return df

def get_params_summary(params, name):
    """
    extracts the statistical descriptors only
    """
    return [name, params[0], params[1], params[2], params[3]]


def generate_pixel_image(coordinates, path_image, path_save, params = None, mask = None, path_save_align = None, combined = False,
                        save = True, WM = True):
    """
    generate_pixel_image overlays the ROIs onto the greyscale images for square images
    
    Parameters
    ----------
    coordinates : list of int
        [x_min, x_max, y_min, y_max] for the ROI 
    path_image : str
        the path to the greyscale image
    path_save : str
        the path to the folder in which the overlaid image should be stored
    mask : array of shape (516, 388)
        the mask of the ROI
    path_save_align, save, params - not used anymore
    combined : boolean
        indicates if we are selecting automatically multiple squares - option removed
    WM : boolean
        indicates if we are working with WM or GM (default : True, changes the color of the ROI borders)
    
    
    Returns
    ----------
    the overlay of the ROIs onto the greyscale image
    """
    if WM:
        val = 0
    else:
        val = 255
        
    if type(mask) == np.ndarray:
        masked = True
    else:
        [x_min, x_max, y_min, y_max] = coordinates
        masked = False
    
    im = Image.open(path_image)
    imnp = np.array(im)
    imnp_mask = np.array(im)
        
    if masked:
        for idx_x, x in enumerate(imnp):
            
            min_idx_x = max(idx_x - 1, 0)
            max_idx_x = min(idx_x + 1, len(mask) - 1)
            
            for idx_y, y in enumerate(x):
                if mask[idx_x][idx_y] == 0 :
                    pass
                
                else:
                    
                    min_idx_y = max(idx_y - 1, 0)
                    max_idx_y = min(idx_y + 1, len(mask[0]) - 1)
                    
                    mask_idx_y = mask[:, idx_y]
                    
                    if mask_idx_y[min_idx_x] == 0:
                        imnp[idx_x][idx_y] = val
                    elif mask_idx_y[max_idx_x] == 0:
                        imnp[idx_x][idx_y] = val
                        
                    else:
                        mask_idx_x = mask[idx_x, :]
                        if mask_idx_x[min_idx_y] == 0:
                            imnp[idx_x][idx_y] = val
                        elif mask_idx_x[max_idx_y] == 0:
                            imnp[idx_x][idx_y] = val

    else:
        for y, row in enumerate(imnp):
            if y < y_min - 1 or y > y_max + 1:
                pass
            else:
                for x, column in enumerate(row):
                    if x < x_min - 1 or x > x_max + 1:
                        pass
                    else:
                        if x_min - 1 <= x <= x_min + 1 or x_max - 1 <= x <= x_max + 1:
                            imnp[y][x] = val
                        if y_min - 1 <= y <= y_min + 1 or y_max - 1 <= y <= y_max + 1:
                            imnp[y][x] = val
                        if x_min <= x <= x_max and y_min <= y <= y_max:
                            imnp_mask[y][x] = val

    if params:
        square_number, square_size, orientation = params
        all_coordinates = get_each_image_coordinates(coordinates, square_number, square_size, orientation)
        for coordinates in all_coordinates:
            [x_min, x_max, y_min, y_max] = coordinates
            for y, row in enumerate(imnp):
                if y < y_min or y > y_max:
                    pass
                else:
                    for x, column in enumerate(row):
                        if x < x_min or x > x_max:
                            pass
                        else:
                            if x_min <= x <= x_min or x_max <= x <= x_max:
                                imnp[y][x] = val
                            if y_min <= y <= y_min or y_max <= y <= y_max:
                                imnp[y][x] = val
                            else:
                                pass
        
    if type(path_save_align) == str and save:  
        Image.fromarray(imnp).save(path_save_align + path_image.split('/')[-1])
    if not combined:
        Image.fromarray(imnp).save(os.path.join(path_save, 'selection.png'))
    
    return imnp_mask.T   


def get_each_image_coordinates(coordinates, square_number, square_size, orientation):
    """
    get_each_image_coordinates was used when multiple squares were selected automatically - the option is not present anymore
    """
    if type(square_size) == int:
        square_size_horizontal = square_size
        square_size_vertical = square_size
    else:
        square_size_horizontal, square_size_vertical = square_size
        
    coordinates_all = []
    x_min, x_max, y_min, y_max = None, None, None, None
    if orientation == 'horizontal':
        for i in range(square_number):
            x_min = coordinates[0] + i*square_size_horizontal
            x_max = coordinates[0] + (i+1)*square_size_horizontal
            y_min = coordinates[2] 
            y_max = coordinates[3]
            coordinates_all.append([x_min, x_max, y_min, y_max])
    else:
        for i in range(square_number):
            x_min = coordinates[0]
            x_max = coordinates[1]
            y_min = coordinates[2] + i*square_size_vertical
            y_max = coordinates[2] + (i+1)*square_size_vertical
            coordinates_all.append([x_min, x_max, y_min, y_max])
    return coordinates_all


def generate_summary_file_series(path_output, square_number, WM):
    """
    generate_summary_file_series is used to create the summary table reporting the statistical metrics of the polarimetric parameters for a single ROI

    Parameters
    ----------
    path_output : str
        the path to the folder in which the results should be outputted
    square_number : int  
        the number of square analyzed (in this case 1)
    """
    summaries_generated_fn = []
    for idx in range(square_number):
        try:
            summaries_generated_fn.append([os.path.join(path_output, 'summary.csv'), idx])
        except:
            pass
        
    summaries_generated = []
    for file in summaries_generated_fn:
        df = pd.read_csv(file[0])
        fn = []
        for i in range(len(df)):
            fn.append(file[1])
        df['square_number'] = fn
        summaries_generated.append(df)

    if summaries_generated:
        result = pd.concat(summaries_generated)
        result = result.sort_values(['square_number', 'parameter']).set_index(['square_number', 'square_size', 'parameter'])
        if WM:
            result.to_csv(os.path.join(path_output.split('WM')[0], 'WM_summaries.csv'))
            result.to_excel(os.path.join(path_output.split('WM')[0], 'WM_summaries.xlsx'))
        else:
            result.to_csv(os.path.join(path_output.split('GM')[0], 'GM_summaries.csv'))
            result.to_excel(os.path.join(path_output.split('GM')[0], 'GM_summaries.xlsx'))
            
