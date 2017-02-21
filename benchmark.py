"""
Implementation of a benchmark system based on dictionary look-up.
Algorithm:
    For each test image, perform normalizd dot-product with every training image
    	normalized dot product = x dot y / ||x|| / ||y||
    Find the one with highest dot-product value
    Look up the one's angle as predicted angle for test image 
    Compute the mean absolute error
"""
import numpy as np
import sys, os
import dictionary as d
import pdb
import time
import numpy.linalg as la
import theano
import theano.tensor as T
#def dotProd(a,b):
    # a: test_img_size x img_dim
    # b: img_dim x train_img_size 
    # output: test_img_size x train_img_size
#    return np.dot(a,b)/la.norm(a)/la.norm(b,axis=0)

#def mae(a,b):
    # a,b: two np arrays with the same length
#    return np.mean(np.absolute(a - b))

seed = 1
nb_train = 1000
nb_test = 100

startT = time.time()

print 'Get data...'
dic = d.Dictionary()
(train_X, train_y), (test_X, test_y), (valid_X, valid_y) = dic.get_normalized_data(seed,nb_train,nb_test)

train_X = np.reshape(train_X,(nb_train,-1)).T.astype('float32')
test_X = np.reshape(test_X,(nb_test,-1)).astype('float32')

print '\tshape of test X: ',np.shape(test_X)
print '\t\texpecting: (%d, 3600)'%nb_test
print '\tshape of train X: ',np.shape(train_X)
print '\t\texpecting: (3600, %d)'%nb_train
print 'Finished, used %s seconds.\n'%(time.time() - startT) 

# Implement dot-prod with Theano
x1 = T.fmatrix()
x2 = T.fmatrix()
x1_norm = T.sqrt(T.nlinalg.diag(T.dot(x1,x1.T))) # nb_test * 1
x2_norm = T.sqrt(T.nlinalg.diag(T.dot(x2.T,x2))) # nb_train * 1
denominator = T.outer(x1_norm,x2_norm) # nb_test * nb_train

d = T.dot(x1,x2) / denominator 

print 'Using Theano, compile func. dotImg'
dotImg = theano.function(
        inputs = [x1,x2],
        outputs = d,
        allow_input_downcast = True)

print '\tstart calculating...'
startT = time.time()

dot_matx = dotImg(test_X, train_X)

print '\tshape of dot product result: ',np.shape(dot_matx)
print '\t\t\texpecting: (%d, %d)'%(nb_test,nb_train)
print 'Finished calculating dot product, used %s secconds.'%(time.time()-startT)

#dotmat = dotProd(test_X,train_X)

# Implement dot-prod with Theano
dd = T.fmatrix()
ag = T.argmax(dd,axis=1)

print 'Using Theano, compile func. argmaxIdx'
argmaxIdx = theano.function(
        inputs = [dd],
        outputs = ag,
        allow_input_downcast = True)

print '\tstart calculating...'
best_idx = argmaxIdx(dot_matx)

#pdb.set_trace()

print '\tshape of best index result: ',np.shape(best_idx)
print '\t\t\texpecting: %d'%nb_test

# Implement MAE with Theano
y1 = T.fvector()
y2 = T.fvector()
e = T.mean(T.abs_(y1-y2))

print 'Using Theano, compile func. maeImg'
maeImg = theano.function(
        inputs = [y1,y2],
        outputs = e,
        allow_input_downcast = True)

pred_y = np.reshape(train_y[best_idx],(nb_test,))
test_y = np.reshape(test_y,(nb_test,))

print "\tStart calculating..."
startT = time.time()

mae = maeImg(test_y,pred_y)

print 'Finished calculating MAE, used %s secconds.'%(time.time()-startT)
print "Benchmark system, MAE = %f"%mae

