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
from dictionary import *
import pdb
import time
import numpy.linalg as la
import theano
import theano.tensor as T
import math
from compute_disorientation import *




data_path = 'training-data'
train_file1 = 'EBSDDictionary_100.h5'
train_file2 = 'EBSDDictionary_50.h5'
test_file = 'EBSDrandom.h5'
train_files = [train_file1, train_file2]

#train_files = [train_file1]


def run_benchmark(target_id=0):
    seed = 8484
    nb_train = 300000
    nb_test = 30000
    batch_size = 1000
    batch_size2 = 100

    print 'Current target id is: ', target_id
    startT = time.time()
    print '\nGet data...'
    dic = Dictionary(data_path, train_files, test_file)
    (train_X, train_y), (test_X, test_y), (valid_X, valid_y) = dic.get_normalized_data(seed,nb_train,nb_test,label_idx=('eu',target_id), target_id=target_id)

    nb_train = train_X.shape[0]
    nb_test = test_X.shape[0]

    train_X = np.reshape(train_X,(nb_train,-1)).T.astype('float32')
    test_X = np.reshape(test_X,(nb_test,-1)).astype('float32')

    print '\tshape of test X:  ',np.shape(test_X)
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

    print 'Compile Theano function dotImg ...'
    dotImg = theano.function(
            inputs = [x1,x2],
            outputs = d,
            allow_input_downcast = True)

    # Implement dot-prod with Theano
    dd = T.fmatrix()
    ag = T.argmax(dd,axis=1)

    print 'Compile Theano function argmaxIdx ...'
    argmaxIdx = theano.function(
            inputs = [dd],
            outputs = ag,
            allow_input_downcast = True)

    print '\tstart calculating...'
    startT = time.time()

    N = nb_test
    N2 = nb_train
    best_idx = None
    print '\tbatch size for left matrix %d, total %d'%(batch_size,N)
    print '\tbatch size for right matrix %d, total %d'%(batch_size2,N2)
    for start,end in zip(range(0,N+1,batch_size), range(batch_size,N+1,batch_size)):
        print '\t\tLeft matrix at --- ',start,end
        dot_matx = None
        for start2,end2 in zip(range(0,N2+1,batch_size2), range(batch_size2,N2+1,batch_size2)):
            #print '\t\t\tRight matrix at --- ',start2,end2
            ins = [test_X[start:end], train_X[:,start2:end2]]
            dot_matx_ = dotImg(*ins)
            if dot_matx is None:
                dot_matx = dot_matx_
            else:
                dot_matx = np.hstack((dot_matx,dot_matx_))
            #print '\t\t\tshape of dot product matrix: ',np.shape(dot_matx)
        best_idx_ = argmaxIdx(dot_matx)
        if best_idx is None:
            best_idx = best_idx_
        else:
             best_idx = np.hstack((best_idx,best_idx_))

        print '\t\tshape of best index matrix: ',np.shape(best_idx)

    #print '\tshape of dot product result: ',np.shape(dot_matx)
    #print '\t\t\texpecting: (%d, %d)'%(nb_test,nb_train)
    print 'Finished calculating dot product & index matrix, used %s secconds.\n'%(time.time()-startT)

    #dotmat = dotProd(test_X,train_X)


    #print '\tstart calculating...'
    #startT = time.time()

    #pdb.set_trace()

    print 'End shape of best index result: ',np.shape(best_idx)
    print '\texpecting: %d'%nb_test

    startT = time.time()
    # Implement MAE with Theano
    y1 = T.fvector()
    y2 = T.fvector()
    e = T.mean(T.abs_(y1-y2))

    print 'Compile Theano function maeImg ...'
    maeImg = theano.function(
            inputs = [y1,y2],
            outputs = e,
            allow_input_downcast = True)

    se = T.mean((y1 - y2)**2)

    mseImg = theano.function(
        inputs=[y1, y2],
        outputs=se,
        allow_input_downcast=True)

    #print 'train_y', train_y.shape, best_idx.shape, nb_test, test_y.shape
    pred_y = np.reshape(train_y[best_idx],(nb_test,))
    test_y = np.reshape(test_y,(nb_test,))

    pred_y = dic.get_original_labels(pred_y)
    test_y = dic.get_original_labels(test_y)
    print 'Testing labels: ', np.mean(test_y), np.median(test_y), np.std(test_y)

    print "\tStart calculating..."

    mae = maeImg(test_y,pred_y)

    mse = mseImg(test_y, pred_y)

    print 'Finished calculating MAE, used %s secconds.'%(time.time()-startT)
    print "Benchmark system, MAE = %f MSE= %f\n"%(mae, mse)
    return  pred_y, test_y, mae, mse

if __name__=="__main__":
    target_ids = [0,1,2]
    maes = []
    mses = []
    maeds = []
    mseds = []
    predicted = None
    actual = None
    for t_id in target_ids:
        pred_y, test_y, mae, mse = run_benchmark(t_id)
        print 'For target_id ', t_id, ' mae is ', mae
        maes.append(mae)
        maeds.append(mae * 180 / math.pi)
        mses.append(mse)
        mseds.append(mse * ((180./math.pi)**2))
        pred_y = np.reshape(pred_y, newshape=(-1,1))
        test_y = np.reshape(test_y, newshape=(-1,1))
        if predicted is None:
            predicted = pred_y
            actual = test_y
        else:
            predicted = np.concatenate((predicted, pred_y), axis=1)
            actual = np.concatenate((actual, test_y), axis=1)
    print 'predicted shape', predicted.shape
    print 'actual shape', actual.shape

    print 'all maes are: ', maes
    print 'all mses are: ', mses

    print 'all maes (in degree are ) are: ', maeds
    print 'all mses (in degree are ) are: ', mseds

    print 'computing disorientation'
    disorientation = compute_disorientations(predicted, actual, is_degree=False)
    print 'mean disorientation is: ', disorientation
