# FreezingEffect

[![License](https://img.shields.io/pypi/l/FreezingEffect.svg?color=green)](https://github.com/RomGr/FreezingEffect/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/FreezingEffect.svg?color=green)](https://pypi.org/project/FreezingEffect)
[![Python Version](https://img.shields.io/pypi/pyversions/FreezingEffect.svg?color=green)](https://python.org)
[![CI](https://github.com/RomGr/FreezingEffect/actions/workflows/ci.yml/badge.svg)](https://github.com/RomGr/FreezingEffect/actions/workflows/ci.yml)


# Effects of freezing on polarimetric properties of brain tissue

by
Romain Gros, Omar Rodrıguez-Nunez, Leonard Felger, Stefano Moriconi, Richard McKinley, Angelo Pierangelo, Tatiana Novikova, Erik Vassella, Philippe Schucht, Ekkehard Hewer and Theoni Maragkou

This repository contains the source code for the paper: Pathology-Guided Quantification of Polarimetric Parameters in Brain Tumors. The manuscript is currently in preparation.

The global study aimed to quantify the polarimetric parameters of neoplastic brain tissue. This part specifically aims at examining the effect of freezing and cryosectioning, an important part of the pathology protocol used to gather pathology diagnosis for the samples considered in our study.


## Abstract

This paper presents a groundbreaking exploration of polarimetric parameters in brain tissue, differentiating between healthy and neoplastic regions. For the first time, we systematically quantified these polarimetric characteristics, shedding light on their potential significance in brain tumor classification. Additionally, we introduced a novel method to correlate polarimetric data with pathology data with a very limited impact on the tissue polarimetric properties, serving as a robust ground truth for our study. Our findings revealed distinct variations in depolarization and linear retardance, particularly noteworthy in the white matter region of tumor tissue, while no significant differences were observed in grey matter. This differential behavior underscores the importance of understanding tissue-specific responses to polarimetry. A significant discovery in our study was the pronounced randomization of the azimuth of the optical axis. This observation holds substantial promise for future developments in machine learning (ML) algorithms aimed at classifying brain tumors. The potential of this newly uncovered parameter to enhance tumor classification accuracy is a pivotal step forward in brain tumor research. Our work sets the stage for the refinement and optimization of ML-based tumor classification algorithms, offering a promising avenue for improving the diagnosis and treatment of brain tumors.


## Software implementation

This GitHub folder documents all the source code used to generate the results and figures relative to the analysis of the effect of the pathology protocol on the polarimetric properties of the tissue. The code is written in Python 3.10.

The source code of the repository is in the `src/freezingeffect` folder.

The [Jupyter notebooks](http://jupyter.org/) were used to generate the results and figures for the paper: `selection_of_ROIs.ipynb` and `get_mask_figure.ipynb`.

The data used in this study should be copied from [this link](to be changed), and placed in the `data` folder. Results generated by the code are saved in `results`.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/RomGr/FreezingEffect.git


## Selection of ROIs

The first juypter notebook can be run to select automatically 15 ROIs within the white matter, and 15 ROIs for each series of measurement located in the folder `data`. CAUTION: running the notebook erase the previously generated results.
Afterwards, the statistical analyses for the evolution of the means and the fold changes is also performed.
The results generated (mainly excel tables), can be found in the folder `results/comparaison`.
The plots for the manuscript were created using Graphpad Prism.

## Get mask figure

The second juypter notebook allows to visualize the distributions of several ROIs, and to determine wether or not these distributions are normal. The output for this notebook can be found in `results/histograms`.


## Parameter comparison

The third juypter notebook, which should be run after the selection of the ROIs, was used to perform the statistical analyses for the evolution of the means and the fold changes. The results generated (mainly excel tables), can be found in the folder `results/comparaison`. The plots for the manuscript were created using Graphpad Prism.


## Evalutation border zone

The last juypter notebook, independent for the first two ones, allows to reproduce the results for the evalutation of the uncertainty region in between white and grey matter, and its evolution following formalin fixation. The results generated can be found in the folders `results/fixed` and `results/fresh`. The plots for the manuscript were created using python and are available directly in the `results` folders.


## Data

Three subfolders can be found in the `data` folder:
1. `fixation_over_time`: contains the measurements for the section performed at different time points
2. `fresh`: contains the measurements for the fresh section
3. `fixed`: contains the measurements for the fixed section performed 24 hours after the fresh ones


## License

All source code is made available under a MIT license. See `LICENSE` for the full license text.
