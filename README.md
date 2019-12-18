# EBSD-indexing

This software contains an implementation of a deep learning model (Convolutional Neural Network) for predicting orientation angles from Electron backscatter diffraction (EBSD)images in material science.
It contains modules for doing grid search on hyperparameter space of deep learning models and also module that resulted in best accuracy for our dataset (EBSD dataset from Prof. Marc De Graef in CMU).
It also contains modules for doing benchmark comparison with other machine learning models (using 1-NN clustering).


## Installation Requirements

System Requirement
==================
Tested on Ubuntu 14.04.2 LTS and Mac OSX 10.11.4.

Python requirement
==================
Pyton 2.7
Requires deuNet library-  a wrapper deep learning library on top of Theano, from CUCIS lab at Northwestern University.
Requires models- numpy, scipy and h5py (can be installed using pip: `pip install numpy`)

If you want to run using GPU, please make sure you have CUDA installed. For example, to run using GPU on superbox in CUCIS lab, you need to change your add following lines to your .bashrc file:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64
export PATH=$PATH:/usr/local/cuda/bin


Deep Learning Framework
=======================
Theano

Package Contents
================
`benchmark.py` - implementation of a benchmark system 1-NN method based on dictionary look-up.
Algorithm:
`benchmark_batch.py` - benchmark with batch sizes rather than individual datapoint to speed up calculation
`cnn_grid_search.py` - grid search in hyperparameter space for best search
`cnn_further_grid_search.py` - modified grid search in hyperparameter space with different set of hyperparameters
`dictionary.py` - data loading and preprocessing for labeling and benchmarking
`marc_best_perf_sys.py` - best performing model for training on our dataset from CMU.
`marc_best_perf_sys_pred.py` - best performing model on our dataset used for prediction of testset

Input Format
============
It expects training dataset as a collection of 60x60 EBSD images (can be tuned for other image dimension) in one file with orientation angles in different formats in other files. All the data preprocessing and loading for the deep learning model is contained in the model file itself. The "correct" way to preprocess target value (y) for CNN training is:
    - convert it from degree (0~360) to radians (0~2pi)
    - use this radians value to calculate loss

However in this documented best-performed program, we
    - we did not do degree-radian conversion
    - degree is normalized into 0~1

We provide a sample dataset in the ‘MarcData’ folder - datafile is 'dictionary-333227-60x60.data' that contains 333227 60x60 images, with orientation angles in 5 representations (Euler, Cubochoric, Homochoric, Quaternion and Rodrigues) with multiple (3 or 4) dimensions in 5 different text files.

Output Format
=============
All the outputs are saved inside logs folder. It will log the model architecture and description of optimizer, along with training epoch results. We provided a sample log file - ‘1st_angle.log’.

To Run
======
Type the following command:
`python marc_best_perf_sys.py`

The training file and all parameters are hard coded in the code.

## Citation

D. Jha, S. Singh, R. Al-Bahrani, W.-keng Liao, A. Choudhary, M. De Graef, and A. Agrawal, "Extracting "Extracting grain orientations from ebsd patterns of polycrystalline materials using convolutional neural networks." Microscopy and Microanalysis 24, no. 5 (2018): 497-502 [DOI:10.1017/S1431927618015131] [<a href="https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4B97FCE81ED02FE7F22148500FD24868/S1431927618015131a.pdf/extracting_grain_orientations_from_ebsd_patterns_of_polycrystalline_materials_using_convolutional_neural_networks.pdf">PDF</a>].


## Questions/Comments

email: dipendra.jha@eecs.northwestern.edu or ankitag@eecs.northwestern.edu</br>
Copyright (C) 2019, Northwestern University.<br/>
See COPYRIGHT notice in top-level directory.


## Funding Support

This work is supported primarily by the AFOSR MURI award FA9550-12-1-0458. Partial support is also acknowledged from the following grants: NIST award 70NANB14H012; NSF award CCF-1409601; DOE awards DE-SC0007456 and DE-SC0014330.
