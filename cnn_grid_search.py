"""
Perform grid-search of hyperparameters in CNN training of Marc's problem
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
local_size = 20
hypergrid = {'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta'], \
             'learning_rate': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], \
             'momentum': [0.5, 0.7, 0.9, 0.95, 0.99], \
             'lr_decay': [1e-2, 1e-3]};

# parameters fixed
nb_iter = 1000  # number of iterations to train, for hyper parameter search
batch_size = 30
w_scale = 1e-2
nesterov = True
rho = 0.9
reg_W = 0.

checkpoint_fn = 'trained_models/.trained_muri_cnn_' + str(
    local_size) + 'x' + str(local_size) + '_grid_search.h5'
log_fn = 'logs/cnn_' + str(local_size) + 'x' + str(
    local_size) + '_grid_search.log'

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
    for learning_rate in hypergrid['learning_rate']:
        for momentum in hypergrid['momentum']:
            for lr_decay in hypergrid['lr_decay']:
                for optim in hypergrid['optimizer']:
                    # NN architecture
                    model = NN(checkpoint_fn, log_fn)

                    model.add(Convolution2D(32, 1, local_size, local_size,
                                            border_mode='full',
                                            init='glorot_uniform',
                                            activation='relu', reg_W=reg_W))
                    model.add(Convolution2D(32, 32, local_size, local_size,
                                            border_mode='valid',
                                            init='glorot_uniform',
                                            activation='relu', reg_W=reg_W))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(Dropout(0.25, uncertainty=False))

                    model.add(Convolution2D(64, 32, local_size, local_size,
                                            border_mode='full',
                                            init='glorot_uniform',
                                            activation='relu', reg_W=reg_W))
                    model.add(Convolution2D(64, 64, local_size, local_size,
                                            border_mode='valid',
                                            init='glorot_uniform',
                                            activation='relu', reg_W=reg_W))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(Dropout(0.25, uncertainty=False))

                    model.add(Flatten())
                    model.add(AffineLayer(64 * 15 * 15, 256, activation='relu',
                                          reg_W=reg_W, init='glorot_uniform'))
                    model.add(Dropout(0.5, uncertainty=False))
                    model.add(
                        AffineLayer(256, 1, activation='linear', reg_W=reg_W,
                                    init='glorot_uniform'))

                    # Compile NN
                    print ('Parameters in grid search: \n' +
                           '\tlocal_size = %s\n' +
                           '\toptimizer = %s\n' +
                           '\tlearning_rate = %s\n' +
                           '\tmomentum = %s\n' +
                           '\tlr_decay = %s\n') % (
                          str(local_size), optim, str(learning_rate),
                          str(momentum), str(lr_decay))
                    print ('Parameters fixed: \n' +
                           '\tbatch_size = %s\n' +
                           '\tnesterov = %s\n' +
                           '\trho = %s\n' +
                           '\treg_W = %s\n') % (
                          str(batch_size), str(nesterov), str(rho), str(reg_W))

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
                          local_size, local_size, local_size, local_size,
                          local_size, local_size, local_size, local_size,
                          optim)

                    print 'Compile CNN ...'
                    model.compile(optimizer=optim, loss='mean_absolute_error',
                                  class_mode="regression",
                                  reg_type='L2', learning_rate=learning_rate,
                                  momentum=momentum,
                                  lr_decay=lr_decay, nesterov=nesterov, rho=rho)

                    # Train NN
                    model.fit(train_X, train_y, valid_X, valid_y,
                              batch_size=batch_size, stopIter=nb_iter,
                              verbose=True)

                # Test NN
                # model.get_test_accuracy(test_X, test_y)
