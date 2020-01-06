import numpy as np
import h5py, os, sys

"""
# Dictionary: the dictionary is basically a lookup table that contains pairs of images and their angles.
# Dictionary size is 333227. Image dimension is 60 x 60. 
# Angles have 5 representations, independent (and transformable) of each other. each representation contains
  multiple (3 or 4) dimensions, each would be modeled separately. 
    1. Euler angles
    2. Cubochoric
    3. Homochoric
    4. Quaternion
    5. Rodrigues

# In our learning system setup, the image is feature signal and the orientation angle is prediction label.

Updated by Rosanne Liu 08-21-2015
"""

class Dictionary(object):
    """
    Dictionary is the collection of images, saved in binary format.
    """
    def __init__(self,data_path, train_file, test_file):
        # get normalization factor and bias depending on the label case
        # to normalize, y -> norm_factor * x + norm_bias
        # case 'eu'
        ##  map to 0~1 
        ##  norm_factor = 1/(max_x - min_x) 
        ##  norm_bias = - min_x/(max_x - min_x) 
        # cases 'cu', 'ho', 'qu', 'ro'
        ##  (map to -1~1)
        ##  norm_factor = 2/(max_x - min_x); norm_bias = - (max_x + min_x)/(max_x - min_x) 
        self.norm_factor = {
                ('cu',0): 1/0.5,  # -0.429006 ~ 0.429006
                ('cu',1): 1/0.5,  # -0.429006 ~ 0.429006
                ('cu',2): 1/0.5,  # -0.429006 ~ 0.429006
                ('eu',0): 1./360.,  # 0 ~ 360
                ('eu',1): 1./60.,   # 0 ~ 60.589
                ('eu',2): 1./360.,  # 0 ~ 360
                ('ho',0): 1./0.4,  # -0.3872 ~ 0.3872
                ('ho',1): 1./0.4,  # -0.3872 ~ 0.3872
                ('ho',2): 1./0.4,  # -0.3872 ~ 0.3872
                ('qu',0): 1./14.,    # 0.856244 ~ 1.0
                ('qu',1): 1./0.4,  # -0.380894 ~ 0.380894
                ('qu',2): 1./0.4,  # -0.380894 ~ 0.380894
                ('ro',0): 1.,    # -1.0 ~ 1.0
                ('ro',1): 1.,    # -1.0 ~ 1.0 
                ('ro',2): 1.,    # -1.0 ~ 1.0 
                }

        self.norm_bias = {
                ('cu',0): 0,  # -0.429006 ~ 0.429006
                ('cu',1): 0,  # -0.429006 ~ 0.429006
                ('cu',2): 0,  # -0.429006 ~ 0.429006
                ('eu',0): 0,  # 0 ~ 360
                ('eu',1): 0,   # 0 ~ 60.589
                ('eu',2): 0,  # 0 ~ 360
                ('ho',0): 0,  # -0.3872 ~ 0.3872
                ('ho',1): 0,  # -0.3872 ~ 0.3872
                ('ho',2): 0,  # -0.3872 ~ 0.3872
                ('qu',0): -13,    # 0.856244 ~ 1.0
                ('qu',1): 0,  # -0.380894 ~ 0.380894
                ('qu',2): 0,  # -0.380894 ~ 0.380894
                ('ro',0): 0,    # -1.0 ~ 1.0
                ('ro',1): 0,    # -1.0 ~ 1.0 
                ('ro',2): 0,    # -1.0 ~ 1.0 
                }
        self.load_data(data_path, train_file, test_file)


    def load_data(self,data_path, trainfiles, testfile):
        """
        load image data from binary file into an array. Shape of array: num_of_images x width x height
        load label data from multiple txt files into a dict.
            # the key of dict is the first two letters identifiable of the representation. 
            # 'cu', 'eu', 'ho', 'qu', 'ro'
         Outputs:
            images: image data, num_of_images * width * height
            labels: dict type structure. e.g. labels['cu']: a (num_of_images * 3) array of anlge values 
        """
        #ori_filename = ["cubochoriclist.txt","eulerlist.txt","homochoriclist.txt","quaternionlist.txt","rodrigueslist.txt"]
        #data = np.fromfile(loc+img_filename, dtype='uint8')
        #images = np.reshape(data,[len(data)/(60*60),60,60],order='C')
        self.datapath = data_path
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
        # print self.train_y
        #print ("Train Keys: %s" % train_keys)
        # for t_k in train_keys:
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
            #print t_k, type(t_k),
            #print test_f[t_k].keys()
            for tff in test_f[t_k].keys():
                try:
                    tffk = test_f[t_k][tff].keys()
                    #print tff, type(tff),
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

        '''
        labels = {}
        for orif in ori_filename:
            f = open(loc+orif, 'r')
            fidx = f.readline().strip('\n')
            if fidx != orif[:2]:
                print "Label file opened wrong!"
            num_samples = int(f.readline())
            labelstrs = f.readlines()
            num_angles = len(labelstrs[0].split())
            
            labels[fidx] = np.zeros((num_samples,num_angles),dtype='float')
            for idx,val in enumerate(labelstrs):
                labels[fidx][idx,:] = [float(number) for number in val.split()]


        return (images,labels)
        '''
        return

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
        np.random.seed(8484)
        np.random.shuffle(random_sequence)
        #print 'type: ', self.train_X, self.train_X.shape, self.train_y, self.train_y.shape
        self.train_X = self.train_X[random_sequence,:]
        self.train_y = self.train_y[random_sequence,:]

    def get_data_r(self, random_seed, train_size,test_size,with_valid='False',valid_size=0,label_idx=('eu',0)):
        """
        separate data into three sets: train, valid and test
        random_seed is used to generate a random non-overlapping separation
        no normalization is performed.
        Inputs:
            random_seed: an integer seed used to draw random samples for each set
            with_valid: when True, valid is a separated set; otherwise valid=test
            train_size: number of training samples, cannot exceed 333227
            test_size: number of test samples, cannot exceed 333227
            valid_size: number of validation samples, cannot exceed 333227
            * Note: train_size + test_size + valid_size cannot exceed 333227. Do not support overlapping now.
            label_idx: two elements. The first indicates which set of label representation is used, corresponding to the keys in labels.
                       choices: 'cu', 'eu', 'ho', 'qu', 'ro'
                       the second indicates the column index in that represetnation. In 'cu' there are 0, 1, 2.
        """
        images,labels = self.load_data()
        self.label_idx = label_idx
        
        if (label_idx[0] in labels.keys()) == False:
            print "Label index class wrong!"
        elif label_idx[1] >= np.shape(labels[label_idx[0]])[1]:
            print "Label index column out of range!"

        X_train = np.zeros((train_size, 60, 60), dtype="uint8")
        y_train = np.zeros((train_size,1), dtype="float")

        X_test = np.zeros((test_size, 60, 60), dtype="uint8")
        y_test = np.zeros((test_size,1), dtype="float")
        
        np.random.seed(random_seed)
        random_sequence = np.arange(333227)
        np.random.shuffle(random_sequence)
        
        train_sequence = random_sequence[:train_size]
        test_sequence = random_sequence[train_size:train_size+test_size]
        
        train_X = images[train_sequence]
        train_y = labels[label_idx[0]][train_sequence, label_idx[1]]
        
        test_X = images[test_sequence]
        test_y = labels[label_idx[0]][test_sequence, label_idx[1]]

        if with_valid == True:
            valid_sequence = random_sequence[train_size+test_size:]
            if len(valid_sequence) != valid_size:
                print "Validation set size wrong!"
            valid_X = images[valid_sequence]
            valid_y = labels[label_idx[0]][valid_sequence, label_idx[1]]
        else:
            valid_X = test_X.copy()
            valid_y = test_y.copy()

        return (train_X, train_y), (test_X, test_y), (valid_X, valid_y)

    def get_original_labels(self, labels):
        return (labels-self.norm_b)/self.norm_f

    def img_show(self):
        pass
        return

    def img_animation(self):
        pass
        return

    def get_normalized_data(self, random_seed,train_size,test_size,with_valid='False',valid_size=0,label_idx=('eu',0),target_id=0):
        """
        Get train, test and valid data sets and have them normalized.
        The normalization of X (grey scale images having pixel valus 0~255) is to divide by 255.
        The normalization of y (angles) are case dependent.
        """
        #(train_X, train_y), (test_X, test_y), (valid_X, valid_y) = self.get_data(random_seed,train_size,test_size,with_valid='False',valid_size=0,label_idx=['eu',0])

        train_X, train_y, test_X, test_y = self.get_data(valid=False, target_id=target_id)
        valid_X, valid_y = test_X.copy(), test_y.copy()

        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')
        valid_X = valid_X.astype('float32')
        train_X /= 255
        test_X /= 255
        valid_X /=255
        
        # get normalization factor  and bias depending on the label case
        # to normalize, y -> norm_factor * x + norm_bias
        self.norm_f = self.norm_factor[label_idx]
        self.norm_b = self.norm_bias[label_idx]

        train_y = train_y.astype('float32')
        test_y = test_y.astype('float32')
        valid_y = valid_y.astype('float32')
        train_y = train_y * self.norm_f + self.norm_b
        test_y = test_y * self.norm_f + self.norm_b
        valid_y = valid_y * self.norm_f + self.norm_b

        return (train_X, train_y), (test_X, test_y), (valid_X, valid_y)

    def get_data(self, valid=True, target_id=0):
        self.shuffle_data()
        if not valid:
            if target_id is None:
                return self.train_X, self.train_y, self.test_X, self.test_y
            else:
                return self.train_X, self.train_y[:,target_id], self.test_X, self.test_y[:,target_id]
        else:
            total_num = self.train_X.shape[0]
            random_sequence = np.arange(total_num)
            np.random.seed(8484)
            np.random.shuffle(random_sequence)
            train_num = int(0.9 * total_num)
            train_seq = random_sequence[:train_num]
            valid_seq = random_sequence[train_num:total_num]
            data = self.train_X
            labels = self.train_y
            print data.shape, labels.shape
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



#if __name__ == "__main__":
#    dic = Dictionary()
#    (train_X, train_y), (test_X, test_y), (valid_X, valid_y) = dic.get_normalized_data(8,300000,33227)
#    print np.shape(train_X), np.shape(train_y)
#    print np.shape(test_X), np.shape(test_y)
#    print np.shape(valid_X), np.shape(valid_y)
#    print np.max(train_X), np.min(train_X)
#    print np.max(train_y), np.min(train_y)
#    print np.max(test_X), np.min(test_X)
#    print np.max(test_y), np.min(test_y)
#    print np.max(valid_X), np.min(valid_X)
#    print np.max(valid_y), np.min(valid_y)
