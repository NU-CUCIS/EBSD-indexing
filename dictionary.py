import numpy as np
import sys, os

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
    def __init__(self):
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

    def load_data(self):
        """
        load image data from binary file into an array. Shape of array: num_of_images x width x height
        load label data from multiple txt files into a dict.
            # the key of dict is the first two letters identifiable of the representation. 
            # 'cu', 'eu', 'ho', 'qu', 'ro'
         Outputs:
            images: image data, num_of_images * width * height
            labels: dict type structure. e.g. labels['cu']: a (num_of_images * 3) array of anlge values 
        """

        loc = "MarcData/"
        img_filename = "dictionary-333227-60x60.data"
        ori_filename = ["cubochoriclist.txt","eulerlist.txt","homochoriclist.txt","quaternionlist.txt","rodrigueslist.txt"]
        data = np.fromfile(loc+img_filename, dtype='uint8')
        images = np.reshape(data,[len(data)/(60*60),60,60],order='C')
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

    def get_data(self, random_seed, train_size,test_size,with_valid='False',valid_size=0,label_idx=('eu',0)):
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

    def img_show(self):
        pass
        return

    def img_animation(self):
        pass
        return

    def get_normalized_data(self, random_seed,train_size,test_size,with_valid='False',valid_size=0,label_idx=('eu',0)):
        """
        Get train, test and valid data sets and have them normalized.
        The normalization of X (grey scale images having pixel valus 0~255) is to divide by 255.
        The normalization of y (angles) are case dependent.
        """
        (train_X, train_y), (test_X, test_y), (valid_X, valid_y) = self.get_data(random_seed,train_size,test_size,with_valid='False',valid_size=0,label_idx=['eu',0])
        
        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')
        valid_X = valid_X.astype('float32')
        train_X /= 255
        test_X /= 255
        valid_X /=255
        
        # get normalization factor  and bias depending on the label case
        # to normalize, y -> norm_factor * x + norm_bias
        norm_f = self.norm_factor[label_idx]
        norm_b = self.norm_bias[label_idx]

        train_y = train_y.astype('float32')
        test_y = test_y.astype('float32')
        valid_y = valid_y.astype('float32')
        train_y = train_y * norm_f + norm_b
        test_y = test_y * norm_f + norm_b
        valid_y = valid_y * norm_f + norm_b

        return (train_X, train_y), (test_X, test_y), (valid_X, valid_y)

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
