"""
Perform grid-search of hyperparameters in CNN training of Marc's problem
Further the search based on the observation of results from cnn_grid_search.py
"""
import numpy as np
import sys

sys.path.append("../deuNet/")

from deuNet.utils import np_utils
from deuNet.datasets import cifar10
from deuNet.models import NN
from deuNet.layers.core import AffineLayer, Dropout
from deuNet.layers.convolutional import Convolution2D, Flatten, MaxPooling2D

import pdb

# parameters in grid search 
# local_size = [9, 11, 15, 20, 25]
local_size = 11
optim_grid = ['SGD', 'RMSprop', 'Adadelta']

# parameters fixed
nb_iter = 1000  # number of iterations to train, for hyper parameter search
batch_size = 30
w_scale = 1e-2
reg_W = 0.

# checkpoint_fn = 'trained_models/.trained_muri_cnn_'+str(local_size)+'x'+str(local_size)+'_grid_search.h5'
# log_fn = 'logs/cnn_'+str(local_size)+'x'+str(local_size)+'_grid_search.log'

datapath = "/raid/rll943/MarcData/"


# get the k-th image from a continuous sequence space
def get_image(k, data):
    img = np.reshape(data[3600 * k:3600 * (k + 1)], [60, 60], order='C')
    return img


# load data
def load_data(seed):
    np.random.seed(seed)

    filename = "dictionary-333227-60x60.data"
    targetname = "eulerangles.txt"

    num_test_samples = 30000
    num_train_samples = 300000

    X_train = np.zeros((num_train_samples, 60, 60), dtype="uint8")
    y_train = np.zeros((num_train_samples, 3), dtype="float")

    X_test = np.zeros((num_test_samples, 60, 60), dtype="uint8")
    y_test = np.zeros((num_test_samples, 3), dtype="float")

    random_sequence = np.random.randint(0, 333227,
                                        size=num_train_samples + num_test_samples)
    train_sequence = random_sequence[:num_train_samples]
    test_sequence = random_sequence[num_train_samples:]

    data = np.fromfile(datapath + filename, dtype='uint8')
    f = open(datapath + targetname, 'r')
    f.readline()
    f.readline()
    labels = f.readlines()

    for idx, val in enumerate(train_sequence):
        img = get_image(val, data)
        X_train[idx, :, :] = img
        y_train[idx, :] = [float(number) for number in labels[val].split()]

    for idx, val in enumerate(test_sequence):
        img = get_image(val, data)
        X_test[idx, :, :] = img
        y_test[idx, :] = [float(number) for number in labels[val].split()]

    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":

    (train_X, train_y), (test_X, test_y) = load_data(1984)
    valid_X, valid_y = test_X, test_y

    # normalize labels
    target_id = 0  ## there are in total 3 dimensions of targets
    norm_factor = np.array([360, 60, 360])
    train_y = train_y[:, target_id].reshape((len(train_y), 1)).astype('float32')
    valid_y = valid_y[:, target_id].reshape((len(valid_y), 1)).astype('float32')
    test_y = test_y[:, target_id].reshape((len(test_y), 1)).astype('float32')
    train_y /= norm_factor[None, target_id]
    valid_y /= norm_factor[None, target_id]
    test_y /= norm_factor[None, target_id]

    # normalize inputs
    train_X = train_X.astype("float32")
    valid_X = valid_X.astype("float32")
    test_X = test_X.astype("float32")
    train_X /= 255
    valid_X /= 255
    test_X /= 255

    train_X = train_X.reshape((-1, 1, 60, 60))
    valid_X = valid_X.reshape((-1, 1, 60, 60))
    test_X = test_X.reshape((-1, 1, 60, 60))

    # Perform grid search for each parameter
    learning_rate_list = []
    nesterov_list = []
    lr_decay_list = []
    momentum_list = []
    rho_list = []
    for optim in optim_grid:
        if optim == 'SGD':
            hypergrid = {'learning_rate_list': [5e-4, 1e-3, 5e-3], \
                         'momentum_list': [0.99, 0.999], \
                         'nesterov_list': [True, False], \
                         'lr_decay_list': [1e-3, 1e-4, 1e-5]};
        if optim == 'RMSprop':
            hypergrid = {'learning_rate_list': [1e-04, 5e-04], \
                         'rho_list': [0.9, 0.99, 0.999]}
        if optim == 'Adadelta':
            hypergrid = {'learning_rate_list': [0.1, 0.5, 0.9], \
                         'rho_list': [0.9, 0.99, 0.999]}
        #
        vars().update(hypergrid)

        pdb.set_trace()

        for learning_rate in learning_rate_list:
            for momentum in momentum_list:
                for nesterov in nesterov_list:
                    for lr_decay in lr_decay_list:
                        for rho in rho_list:

                            # NN architecture
                            model = NN()

                            model.add(
                                Convolution2D(32, 1, local_size, local_size,
                                              border_mode='full',
                                              init='glorot_uniform',
                                              activation='relu', reg_W=reg_W))
                            model.add(
                                Convolution2D(32, 32, local_size, local_size,
                                              border_mode='valid',
                                              init='glorot_uniform',
                                              activation='relu', reg_W=reg_W))
                            model.add(MaxPooling2D(pool_size=(2, 2)))
                            model.add(Dropout(0.25, uncertainty=False))

                            model.add(
                                Convolution2D(64, 32, local_size, local_size,
                                              border_mode='full',
                                              init='glorot_uniform',
                                              activation='relu', reg_W=reg_W))
                            model.add(
                                Convolution2D(64, 64, local_size, local_size,
                                              border_mode='valid',
                                              init='glorot_uniform',
                                              activation='relu', reg_W=reg_W))
                            model.add(MaxPooling2D(pool_size=(2, 2)))
                            model.add(Dropout(0.25, uncertainty=False))

                            model.add(Flatten())
                            model.add(AffineLayer(64 * 15 * 15, 256,
                                                  activation='relu',
                                                  reg_W=reg_W,
                                                  init='glorot_uniform'))
                            model.add(Dropout(0.5, uncertainty=False))
                            model.add(AffineLayer(256, 1, activation='linear',
                                                  reg_W=reg_W,
                                                  init='glorot_uniform'))

                            # Compile NN
                            print ('Parameters in grid search: \n' +
                                   '\tfilter_size = %s\n' +
                                   '\toptimizer = %s\n' +
                                   '\tlearning_rate = %s\n' +
                                   '\tmomentum = %s\n' +
                                   '\tnesterov = %s\n' +
                                   '\tlr_decay = %s\n' +
                                   '\trho = %s\n') % (
                                  local_size, optim, learning_rate, momentum,
                                  nesterov, lr_decay, rho)

                            print ('Architecture: \n' +
                                   '\tInput: 60 x 60 x 1\n'
                                   '\tConv 1: %d x %d, depth=32, full mode;\n' +
                                   '\tConv 2: %d x %d, depth=32, valid mode;\n' +
                                   '\tPool 1: (2,2);\n' +
                                   '\tDropout 1: 0.25\n' +
                                   '\tConv 3: %d x %d, depth=64, full mode; \n' +
                                   '\tConv 4: %d x %d, depth=64, valid mode; \n' +
                                   '\tPool 2: (2,2); \n' +
                                   '\tDropout 2: 0.25\n' +
                                   '\tFC: 256;\n' +
                                   '\tDropout: 0.5\n' +
                                   '\tFC: 1;\n' +
                                   '\tOptimizer: %s\n') % (
                                  local_size, local_size, local_size,
                                  local_size,
                                  local_size, local_size, local_size,
                                  local_size,
                                  optim)

                            print 'Compile CNN ...'

                            if optim == 'SGD':
                                model.compile(optimizer=optim,
                                              loss='mean_absolute_error',
                                              class_mode="regression",
                                              reg_type='L2',
                                              learning_rate=learning_rate,
                                              momentum=momentum,
                                              lr_decay=lr_decay,
                                              nesterov=nesterov)
                            else:
                                model.compile(optimizer=optim,
                                              loss='mean_absolute_error',
                                              class_mode="regression",
                                              reg_type='L2',
                                              learning_rate=learning_rate,
                                              rho=rho)
                            # Train NN
                            model.fit(train_X, train_y, valid_X, valid_y,
                                      batch_size=batch_size, stopIter=nb_iter,
                                      verbose=True)

                        # Test NN
                        # model.get_test_accuracy(test_X, test_y)
