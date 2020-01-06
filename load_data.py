import h5py, os, sys
import numpy as np
from sklearn import preprocessing


class LoadData():
    def __init__(self, datapath, trainfiles, testfile):
        self.datapath = datapath
        self.trainfiles = trainfiles
        self.testfile = testfile
        self.train_X = None
        self.train_y = None
        for trainfile in self.trainfiles:
            train_f = h5py.File(os.path.join(self.datapath, trainfile))
            train_keys = train_f.keys()
            ebsd_patterns = train_f['EMData']['EBSDpatterns'][()]
            print ebsd_patterns.shape
            eulerangles = train_f['EMData']['Eulerangles']
            print eulerangles.shape
            train_X = ebsd_patterns
            train_y = eulerangles
            if self.train_X is None:
                self.train_X = np.array(train_X)
                self.train_y = np.array(train_y)
            else:
                self.train_X = np.concatenate((self.train_X, train_X), axis=0)
                self.train_y = np.concatenate((self.train_y, train_y), axis=0)
        #print self.train_y
        #print ("Train Keys: %s" % train_keys)
        #for t_k in train_keys:
        #    print t_k, type(t_k),
        #    print train_f[t_k].keys()
        #    for tff in train_f[t_k].keys():
        #        try:
        #            tffk = train_f[t_k][tff].keys()
        #            print tff, type(tff),
        #            print tffk
        #        except:
        #            pass
        test_f = h5py.File(os.path.join(self.datapath, self.testfile))
        test_keys = test_f.keys()
        for t_k in test_keys:
            #print t_k,type(t_k),
            #print test_f[t_k].keys()
            for tff in test_f[t_k].keys():
                try:
                    tffk = test_f[t_k][tff].keys()
                    #print tff,type(tff),
                    #print tffk
                except:
                    pass
        test_ebsd_patterns = test_f['EMData']['EBSD']['EBSDpatterns'][()]
        test_eulerangles = test_f['EMData']['EBSD']['Eulerangles'][()]
        #print ("Test Keys: %s" % test_keys)
        self.test_X = test_ebsd_patterns
        self.test_y = test_eulerangles
        self.test_X = np.array(self.test_X)
        self.test_y = np.array(self.test_y)
        self.shuffle_data()
        self.analyze_data()

    def analyze_data(self):
        for i in range(3):
            print 'distribution for angle ', i, ': ', np.mean(self.train_y[:, i]), np.std(self.train_y[:, i]), np.mean(
                self.test_y[:, i]), np.std(self.test_y[:, i])
        data = self.train_X
        data = np.reshape(data, (-1,60*60))
        print 'Training: ', np.mean(data), np.median(data), np.std(data)
        data = self.test_X
        data = np.reshape(data, (-1, 60 * 60))
        print 'Testing: ', np.mean(data), np.median(data), np.std(data)

    def shuffle_data(self):
        random_sequence = np.arange(self.train_X.shape[0])
        #np.random.seed(374852)
        np.random.seed(8787)
        np.random.shuffle(random_sequence)
        #print 'type: ', self.train_X, self.train_X.shape, self.train_y, self.train_y.shape
        self.train_X = self.train_X[random_sequence,:]
        self.train_y = self.train_y[random_sequence,:]

    def get_data(self, valid=True, target_id=0):
        self.shuffle_data()
        if not valid:
            return self.train_X, self.train_y, self.test_X, self.test_y, self.test_X, self.test_y
        else:
            total_num = self.train_X.shape[0]
            random_sequence = np.arange(total_num)
            #np.random.seed(374852)
            np.random.seed(8787)
            np.random.shuffle(random_sequence)
            train_num = int(0.95 * total_num)
            train_seq = random_sequence[:train_num]
            valid_seq = random_sequence[train_num:total_num]
            data = self.train_X
            labels = self.train_y
            #print data.shape, labels.shape
            self.train_X = data[train_seq, :]
            self.valid_X = data[valid_seq, :]
            if target_id != None:
                labels = labels[:,target_id]
                self.test_y = self.test_y[:,target_id]
                #train_seq = sorted(train_seq)
                #valid_seq = sorted(valid_seq)
                self.train_y = labels[train_seq]
                self.valid_y = labels[valid_seq]
            else:
                labels1 = labels[:, 0]
                labels2 = labels[:, 1]
                labels3 = labels[:, 2]
                trainy1 = labels1[train_seq]
                trainy2 = labels2[train_seq]
                trainy3 = labels3[train_seq]
                valid1 = labels1[valid_seq]
                valid2 = labels2[valid_seq]
                valid3 = labels3[valid_seq]
                self.train_y = np.column_stack((trainy1, trainy2, trainy3))
                self.valid_y = np.column_stack((valid1, valid2, valid3))
            return self.train_X, self.train_y, self.valid_X, self.valid_y, self.test_X, self.test_y

    def bg_process(self, X, parameter='mean', operation='sub', axis=0):
        if axis==0:
            num_features = reduce(lambda x,y: x*y, X.shape[1:])
            for feat in range(num_features):
                #print feat, 'sum of feat: ', np.sum(X[:,feat:feat+1]), X[:,feat:feat+1].shape
                #print X[:,feat:feat+1]
                param = np.mean(X[:,feat:feat+1])
                if parameter == 'median':
                    param = np.median(X[:,feat:feat+1])
                if operation == 'sub':
                    X[:,feat:feat+1] = X[:,feat:feat+1] - param
                else:
                    if param==0:
                        print 'param is zero'
                        return None
                    X[:,feat:feat+1] = X[:,feat:feat+1]/param
                    if math.isnan(np.sum(X[point:point+1,:])):  print param
                #print param, np.sum(X[:, feat:feat+1])

        else:
            num_points = X.shape[0]
            for point in range(num_points):
                #print point, 'sum of point: ', np.sum(X[point:point+1,:]), X[point:point+1,:].shape
                param = np.mean(X[point:point+1,:])
                if parameter == 'median':
                    param = np.median(X[point:point+1,:])
                #print param
                if operation == 'sub':
                    X[point:point+1,:] = X[point:point+1,:] - param
                else:
                    if param==0:
                        print 'param is zero'
                        return None
                    X[point:point+1,:] = X[point:point+1,:] / param
                    if math.isnan(np.sum(X[point:point+1,:])): print param
                #print np.sum(X[point:point+1,:])
        #print X
        return X

    def preprocess(self, preprocess=True, normalize=False, norm='l2', parameter='mean', operation='sub', axis=1):
        # print 'preprocessing parameters:', X.shape, normalize, norm, parameter, operation, axis
        # print X.shape
        print 'before preprocessing: ',self.train_X[0,0,0], self.test_X[0,0,0], self.train_X.shape, self.test_X.shape
        i=0
        for X in [self.train_X, self.test_X]:
            X_n = np.copy(X)
            X_n = np.reshape(X_n, (X_n.shape[0], -1))
            # print X_n
            # print X_n.shape, np.sum(X_n)
            if normalize:
                X_n = preprocessing.normalize(X_n, norm=norm, axis=axis, copy=False, return_norm=False)
            else:
                X_n = self.bg_process(X_n, parameter, operation, axis=axis)
            X_n = np.reshape(X_n, (X.shape))
            # print X_n.shape, np.sum(X_n)
            if i==0:   self.train_X = np.copy(X_n)
            if i==1:    self.test_X = np.copy(X_n)
            i+=1
        print 'after preprocessing: ',self.train_X[0,0,0], self.test_X[0,0,0]
        print 'shape after preprocessing: ', self.train_X.shape, self.test_X
