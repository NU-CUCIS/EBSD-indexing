"""
Best parameter setting so far of Marc's problem
	9x9 filter size
	batch  = 30
	lr = 1e-03
	momentum = 0.9
"""
import numpy as np
import sys

sys.path.append("/Users/dipendra/Libs//DLibs/")

from deuNet.utils import np_utils
from deuNet.models import NN
from deuNet.layers.core import AffineLayer, Dropout
from deuNet.layers.convolutional import Convolution2D,Flatten,MaxPooling2D

import pdb

batch_size = 10
nb_epoch = 500
learning_rate = 1e-3
w_scale = 1e-2
momentum = 0.9
lr_decay = 1e-6
nesterov = True
rho = 0.5
reg_W = 0.

filter_size = 9

# decrease the lr to lr*lr_drop_rate every epoch_step
lr_drop_rate = 0.5
epoch_step = 100

checkpoint_fn = 'trained_models/.best_perf.h5'
log_fn = 'logs/best_perf.log'

datapath = "MarcData/"

# get the k-th image from a continuous sequence space
def get_image(k,data):
    img = np.reshape(data[3600*k:3600*(k+1)],[60,60],order='C')
    return img

# load data
def load_data(seed):
    np.random.seed(seed)

    filename = "dictionary-333227-60x60.data"
    targetname = "eulerangles.txt"

    num_test_samples = 30000
    num_train_samples = 300000

    X_train = np.zeros((num_train_samples, 60, 60), dtype="uint8")
    y_train = np.zeros((num_train_samples,3), dtype="float")

    X_test = np.zeros((num_test_samples, 60, 60), dtype="uint8")
    y_test = np.zeros((num_test_samples,3), dtype="float")

    random_sequence = np.arange(333227)
    np.random.seed(8484)
    np.random.shuffle(random_sequence)

    train_sequence = random_sequence[:num_train_samples]
    test_sequence = random_sequence[num_train_samples:num_train_samples+num_test_samples]
    if set(test_sequence).intersection(train_sequence):
        raise ValueError("Overlap between train and test sequences!")

    data = np.fromfile(datapath+filename, dtype='uint8')
    f = open(datapath+targetname, 'r')
    f.readline()
    f.readline()
    labels = f.readlines()

    for idx,val in enumerate(train_sequence):
	    img = get_image(val,data)
	    X_train[idx, :, :] = img
	    y_train[idx,:] = [float(number) for number in labels[val].split()]

    for idx,val in enumerate(test_sequence):
	    img = get_image(val,data)
	    X_test[idx, :, :] = img
	    y_test[idx,:] = [float(number) for number in labels[val].split()]
   
    f.close()

    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    
    (train_X, train_y), (test_X, test_y) = load_data(1984)
    valid_X,valid_y = test_X, test_y

    print 'Shape:', valid_X.shape, valid_y.shape, train_X.shape, train_y.shape
    # model one target at a time
    target_id = 0 ## there are in total 3 dimensions of targets
    train_y = train_y[:,target_id].reshape((len(train_y),1)).astype('float32')
    valid_y = valid_y[:,target_id].reshape((len(valid_y),1)).astype('float32')
    test_y = test_y[:,target_id].reshape((len(test_y),1)).astype('float32')
    train_y /= 360
    valid_y /= 360
    test_y  /= 360

    print 'train_X', train_X[0]
    print 'train_y', train_y[0]
    print 'valid_X', valid_X[0]
    print 'valid_y', valid_y[0]
    
    print 'Checkpoint saved at ... %s\nLogs written at ... %s\n'%(checkpoint_fn, log_fn)
    print 'Decrease learning rate to %f every %d epochs\n'%(lr_drop_rate, epoch_step)

    # model three targets togeter
    #train_y = train_y.reshape((len(train_y),3)).astype('float32')
    #valid_y = valid_y.reshape((len(valid_y),3)).astype('float32')
    #test_y = test_y.reshape((len(test_y),3)).astype('float32')

    print "before degree conversion, range of label 0 in train: (%.6f, %.6f)"%(min(train_y[:,0]), max(train_y[:,0]))
    print "before degree conversion, range of label 0 in test: (%.6f, %.6f)"%(min(test_y[:,0]), max(test_y[:,0]))
    #print "before degree conversion, range of label 1 in train: (%.6f, %.6f)"%(min(train_y[:,1]), max(train_y[:,1]))
    #print "before degree conversion, range of label 1 in test: (%.6f, %.6f)"%(min(test_y[:,1]), max(test_y[:,1]))
    #print "before degree conversion, range of label 2 in train: (%.6f, %.6f)"%(min(train_y[:,2]), max(train_y[:,2]))
    #print "before degree conversion, range of label 2 in test: (%.6f, %.6f)"%(min(test_y[:,2]), max(test_y[:,2]))

    #train_y = np.radians(train_y)
    #valid_y = np.radians(valid_y)
    #test_y = np.radians(test_y)
    
    #print "\n"
    #print "Range of label 0 in train after deg-rad conversion: (%.6f, %.6f)"%(min(train_y[:,0]), max(train_y[:,0]))
    #print "Range of label 0 in test after deg-rad conversion: (%.6f, %.6f)"%(min(valid_y[:,0]), max(valid_y[:,0]))
    #print "Expected: (0, %.6f)\n"%(2*np.pi)
    #print "Range of label 1 in train after deg-rad conversion: (%.6f, %.6f)"%(min(train_y[:,1]), max(train_y[:,1]))
    #print "Range of label 1 in test after deg-rad conversion: (%.6f, %.6f)"%(min(valid_y[:,1]), max(valid_y[:,1]))
    #print "Expected: (0, %.6f)\n"%(np.pi/6)
    #print "Range of label 2 in train after deg-rad conversion: (%.6f, %.6f)"%(min(train_y[:,2]), max(train_y[:,2]))
    #print "Range of label 2 in test after deg-rad conversion: (%.6f, %.6f)"%(min(valid_y[:,2]), max(valid_y[:,2]))
    #print "Expected: (0, %.6f)\n"%(2*np.pi)

    #print "### The train/test error in NN represent a degree difference in radians between (0,pi)."
    #print "### multiply by 180/pi to get the difference in degrees. \n"

    # normalize inputs
    train_X = train_X.astype("float32")
    valid_X = valid_X.astype("float32")
    test_X = test_X.astype("float32")
    train_X /= 255
    valid_X /= 255
    test_X  /= 255
    
    train_X = train_X.reshape((-1,1,60,60))
    valid_X = valid_X.reshape((-1,1,60,60))
    test_X = test_X.reshape((-1,1,60,60))

	# NN architecture
    # no pooling
    model = NN(checkpoint_fn,log_fn)
    # original: 60 x 60 x 1
    model.add(Convolution2D(32,1,filter_size,filter_size, border_mode='full',
    		init='glorot_uniform',activation='relu', reg_W=reg_W)) # (60+8) x (60+8) x 32
    model.add(Convolution2D(32,32,filter_size,filter_size, border_mode='valid',
    		init='glorot_uniform',activation='relu', reg_W=reg_W)) # (68-8) x (68-8) x 32
    model.add(MaxPooling2D(pool_size=(2,2))) # 30 x 30 x 32
    model.add(Dropout(0.25, uncertainty=False))
    
    model.add(Convolution2D(64,32,filter_size,filter_size, border_mode='full',
    		init='glorot_uniform',activation='relu', reg_W=reg_W)) # 30 x 30 x 64
    model.add(Convolution2D(64,64,filter_size,filter_size, border_mode='valid',
    		init='glorot_uniform',activation='relu', reg_W=reg_W)) # 30 x 30 x 64
    model.add(MaxPooling2D(pool_size=(2,2))) # 15 x 15 x 64
    model.add(Dropout(0.25,uncertainty=False))
    
    model.add(Flatten())
    model.add(AffineLayer(64*15*15, 256,activation='relu',reg_W=reg_W, init='glorot_uniform'))
    model.add(Dropout(0.5, uncertainty=False))
    model.add(AffineLayer(256, 1,activation='linear',reg_W=reg_W,init='glorot_uniform'))

	# Compile NN
    print ('Parameters: \n' +
		'\tfilter_size = %s\n' +
		'\tbatch_size = %s\n' +
		'\tnb_epoch = %s\n' +
		'\tlearning_rate = %s\n' +
		'\tmomentum = %s\n' +
		'\tlr_decay = %s\n' +
		'\tnesterov = %s\n' +
		'\trho = %s\n' +
		'\treg_W = %s\n')%(filter_size, batch_size, nb_epoch, learning_rate,
					momentum,lr_decay,nesterov,rho, reg_W)
        
    print ('Architecture: \n' +
		'\tInput: 60 x 60 x 1\n'
		'\tConv 1: 9x9, depth=32, full mode; output: 68 x 68 x 32\n' +
		'\tConv 2: 9x9, depth=32, valid mode; output: 60 x 60 x 32\n' +
		'\tPool 1: (2,2); output: 30 x 30 x 32\n' +
		'\tDropout 1: 0.25\n' +
		'\tConv 3: 9x9, depth=64, full mode; output: 38 x 38 x 64\n' +
		'\tConv 4: 9x9, depth=64, valid mode; output: 30 x 30 x 64\n' +
		'\tPool 2: (2,2); output: 15 x 15 x 64\n' +
		'\tDropout 2: 0.25\n' +
		'\tFC 1: 256; output: 256 x 1\n' +
		'\tDropout: 0.5\n' +
		'\tFC 2: 1: output 1\n' +
		'\tLoss func: angular_mean_absolute_error\n' +
		'\tOptimizer: SGD\n')
    
    print 'Compile CNN ...'
    model.compile(optimizer='SGD', loss='angular_mae', class_mode="regression",
        	reg_type='L2', learning_rate = learning_rate, momentum=momentum,
        	lr_decay=lr_decay, nesterov=nesterov, rho=rho)

	# Train NN
    model.fit(train_X, train_y, valid_X, valid_y,
        	batch_size=batch_size, nb_epoch=nb_epoch, verbose=True,
            epoch_step=epoch_step,lr_drop_rate=lr_drop_rate)

	# Test NN
	#model.get_test_accuracy(test_X, test_y)

