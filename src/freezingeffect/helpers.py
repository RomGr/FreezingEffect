import json
import os

def load_parameter_maps():
    """
    load and returns the parameters for the histogram plots

    Returns
    -------
    parameters_map : dict
        the parameters to plot the parameters histograms
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'parameters_map.json')) as json_file:
        data = json.load(json_file)
    return data

def load_histogram_parameters():
    """
    load and returns the parameters for the histogram plots

    Returns
    -------
    parameters_map : dict
        the parameters to plot the parameters histograms
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'histogram_parameters.json')) as json_file:
        data = json.load(json_file)
    return data

def load_param_names_link(inv: bool = False):
    """
    load and returns the parameters for the histogram plots

    Returns
    -------
    parameters_map : dict
        the parameters to plot the parameters histograms
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'polarimetric_parameters_link.json')) as json_file:
        data = json.load(json_file)
    if inv:
        data = {v: k for k, v in data.items()}
    return data

def load_parameters_ROIs():
    """
    load and returns the parameters for the histogram plots

    Returns
    -------
    parameters_map : dict
        the parameters to plot the parameters histograms
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'parameters.json')) as json_file:
        data = json.load(json_file)
    return data

def load_param_names():
    """
    load and returns the name of the files that will be generated during the processing

    Returns
    -------
    filenames : list
        the list of the files generated during the processing
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'polarimetric_parameters.txt')) as f:
        lines = f.readlines()
    f.close()
    for idx, l in enumerate(lines):
        lines[idx] = l.replace('\n', '')
    return lines

import numpy as np

def get_angle(fname):
    with open(fname) as f:
        lines = f.readlines()

    for line in lines:
        if 'TransformParameters ' in line:
            angle_data = line
    angle_data = angle_data.split(' ')[1:5]
    angle = np.arctan(float(angle_data[2])/float(angle_data[0]))
    
    return angle*360/(2*np.pi)