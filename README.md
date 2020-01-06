# EBSD-indexing

This repository provides a deep learning approach to the indexing of Electron Backscatter Diffraction (EBSD) patterns. It contains a deep convolutional neural network architecture to predict crystal orientation from the EBSD patterns. We design a differentiable approximation to the disorientation function between the predicted crystal orientation and the ground truth; the deep learning model optimizes for the mean disorientation error between the predicted crystal orientation and the ground truth using stochastic gradient descent.
It also contains modules for doing benchmark comparison with other machine learning models (using 1-NN clustering).


## Installation Requirements

Pyton 2.7

numpy 1.15.4

TensorFlow 1.12

Theano 1.0.1

h5py 2.9.0

## Source Files

* `benchmark_batch.py` - implementation of a benchmark system 1-NN method with batch sizes based on dictionary look-up.
Algorithm:

* `dictionary.py` - data loading and preprocessing for labeling and benchmarking.

* `load_data.py` - data loading and preprocessing for training the CNN model from the .h5 files.

* `model.py` - code for training CNN model.

* `compute_disorientation.py` - implementation of a differential approximation to the disorientation function between two pairs of orientations using TensorFlow with minibatch processing for speed up.
* `train_utils.py` - utility code for training the model.

* `training-data` - Please follow the README file inside the folder to download the required dataset.

* `sample` - contains log file from running the model on the dataset used in the paper [1].

## To Run

The CNN model can be run by using the following command:

`python model.py`

The training files and test/validation files are hard coded in the code along with the CNN model architecture and hyperparameters. The training dataset is a collection of 60x60 EBSD images with orientation angles in Euler angles. The code contains the required preprocessing and scaling done before the actual training of the model. The expected output log from running the CNN model on the training dataset used in the paper is provided in the `sample` folder.



## Publications

Please cite the following paper if you are using this model and code:

Dipendra Jha, Saransh Singh, Reda Al-Bahrani, Wei-keng Liao, Alok Choudhary, Marc De Graef, and Ankit Agrawal, "Extracting "Extracting grain orientations from ebsd patterns of polycrystalline materials using convolutional neural networks." Microscopy and Microanalysis 24, no. 5 (2018): 497-502 [DOI:10.1017/S1431927618015131] [<a href="https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4B97FCE81ED02FE7F22148500FD24868/S1431927618015131a.pdf/extracting_grain_orientations_from_ebsd_patterns_of_polycrystalline_materials_using_convolutional_neural_networks.pdf">PDF</a>].


## Questions/Comments

email: dipendra.jha@eecs.northwestern.edu or ankitag@eecs.northwestern.edu</br>
Copyright (C) 2019, Northwestern University.<br/>
See COPYRIGHT notice in top-level directory.


## Funding Support

This work is supported primarily by the AFOSR MURI award FA9550-12-1-0458. Partial support is also acknowledged from the following grants: NIST award 70NANB14H012; NSF award CCF-1409601; DOE awards DE-SC0007456 and DE-SC0014330.
