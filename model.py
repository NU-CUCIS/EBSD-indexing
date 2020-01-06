import sys, traceback
from load_data import *
import re, math
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from train_utils import *
import numpy as np
import tensorflow.contrib.slim as slim, time

from compute_disorientation import *


from tensorflow.python import debug as tf_debug

batch_size = 64
num_epochs = 1000
EVAL_FREQUENCY=1000
learning_rate = 1e-4
momentum = 0.95
architecture= 'infile'
optimizer = 'SGD'
reg_type='L2'
reg_W = 0.

patience = 100
filter_size = 3
IMAGE_SIZE = 60
NUM_CHANNELS = 1
stride = 1

# decrease the lr to lr*lr_drop_rate every epoch_step
lr_drop_rate = 0.5
epoch_step = 100
logfile = 'temp_'+get_date_str()+'.txt'
log_folder = os.getcwd()+'/sample'
os.system('mkdir -p '+log_folder)

save_dir = os.path.join(log_folder, logfile.split('.')[0])
os.system('mkdir -p '+save_dir)

conffile = None

checkpoint_fn = os.path.join(log_folder,
							 'checkpoint_'+logfile+'.h5')
log_fn = os.path.join(log_folder, 'nn_'+logfile)
pp_file = os.path.join(log_folder, 'pp_'+logfile)
rr_file = os.path.join(log_folder, 'rr_'+logfile)

rr = Record_Results(rr_file)

if architecture == 'infile':
    architecture = [{'layer_type':'conv', 'num_filters':256, 'input_channels':1, 'filter_size':filter_size, 'border_mode':'same', 'init':'glorot_uniform', 'stride':stride,'activation':'relu', 'reg_W':reg_W},
                    {'layer_type': 'conv', 'num_filters': 256,'stride':stride, 'input_channels': 256, 'filter_size': filter_size, 'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', 'reg_W': reg_W},
                    #{'layer_type': 'maxpool2D', 'pool_size':2},
                    #{'layer_type':'dropout', 'value':0.5},
                    {'layer_type': 'conv', 'num_filters': 512,'stride':stride, 'input_channels': 256, 'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', 'reg_W': reg_W},
                    {'layer_type': 'conv', 'num_filters': 512,'stride':stride, 'input_channels': 512, 'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', reg_W: reg_W},
                    {'layer_type': 'maxpool2D', 'pool_size': 2},
                    #{'layer_type': 'dropout', 'value': 0.5},
                    {'layer_type': 'conv', 'num_filters': 256, 'stride': stride, 'input_channels': 256,
                     'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', 'reg_W': reg_W},
                    {'layer_type': 'conv', 'num_filters': 256, 'stride': stride, 'input_channels': 256,
                     'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', reg_W: reg_W},

                    {'layer_type': 'conv', 'num_filters': 96, 'stride': stride, 'input_channels': 256,
                     'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', 'reg_W': reg_W},
                    {'layer_type': 'conv', 'num_filters': 96, 'stride': stride, 'input_channels': 96,
                     'filter_size': filter_size,
                     'border_mode': 'same', 'init': 'glorot_uniform', 'activation': 'relu', reg_W: reg_W},
                    {'layer_type': 'maxpool2D', 'pool_size': 2},
                    {'layer_type': 'flatten'},
                    {'layer_type': 'fully_connected','num_outputs': 8192,'num_inputs':96 * 15 * 15,'activation':'relu', 'reg_W':reg_W, 'init':'glorot_uniform'},
                    {'layer_type': 'fully_connected', 'num_outputs': 2048, 'num_inputs': 8192,
                     'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform'},

                    {'layer_type': 'fully_connected', 'num_outputs': 1024, 'num_inputs': 2048,
                     'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform','branch':True},
                    {'layer_type': 'fully_connected', 'num_outputs': 256, 'num_inputs': 1024,
                     'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform'},

                    {'layer_type':'fully_connected','num_outputs': 1, 'num_inputs':256, 'activation':'linear', 'reg_W':reg_W, 'init':'glorot_uniform'}
    ]


rr.fprint('Architecture:',architecture)
if conffile is not None:
    rr.fprint('Configuration file is: ' + conffile)

SEED = 66478
def model_slim(data, architecture, train=True):
    i=0
    branch = False
    start_branch = False
    if train:
        reuse = None
    else:
        reuse = True
    nets = {}
    nets[0] = data
    for arch in architecture:
        i +=1
        layer_type = arch['layer_type']
        if arch.has_key('branch') and arch['branch']:
            if not start_branch: start_branch = True
            else: start_branch = False
            branch = True
        else:
            start_branch = False
        if layer_type == 'conv':
            print 'adding cnn layer..', i
            num_filters = arch['num_filters']
            filter_size = arch['filter_size']
            border_mode = 'SAME'
            activation = tf.nn.relu
            if arch.has_key('border_mode'):
                border_mode = arch['border_mode']
            padding=border_mode
            if arch.has_key('padding'):
                padding = arch['padding']
            if arch.has_key('activation'):
                if arch['activation'] == 'sigmoid':
                    activation = tf.nn.sigmoid
            stride = 1
            if arch.has_key('stride'):
                stride = arch['stride']
            weights_initializer = tf.truncated_normal_initializer(stddev=0.05)
            if not branch:
                print 'not branch'
                nets[i] = slim.layers.conv2d(nets[i-1], num_outputs=num_filters,kernel_size=[filter_size, filter_size], weights_initializer=weights_initializer, padding=padding, scope='conv'+str(i), stride=stride, weights_regularizer=slim.l2_regularizer(0.001), reuse=reuse, activation_fn=activation)
            elif branch:
                print 'branch'
                nets[i] = [None, None, None]
                for j in range(3):
                    if start_branch:
                        print 'start branch...',j
                        nets[i][j] = slim.layers.conv2d(nets[i - 1], num_outputs=num_filters,
                                             kernel_size=[filter_size, filter_size],
                                             weights_initializer=weights_initializer, padding=padding, weights_regularizer=slim.l2_regularizer(0.001),
                                             scope='conv' + str(i)+str(j), stride=stride, reuse=reuse, activation_fn=activation)
                    else:
                        print 'not start branch'
                        nets[i][j] = slim.layers.conv2d(nets[i - 1][j], num_outputs=num_filters,
                                                        kernel_size=[filter_size, filter_size],
                                                        weights_initializer=weights_initializer, padding=padding, weights_regularizer=slim.l2_regularizer(0.001),
                                                        scope='conv' + str(i)+str(j), stride=stride, reuse=reuse, activation_fn=activation)

        elif layer_type == 'fully_connected':
            num_outputs = arch['num_outputs']
            activation == tf.nn.relu
            if arch['activation'] == 'sigmoid':
                activation = tf.nn.sigmoid
            elif arch['activation'] =='linear':
                activation = None
            print 'adding fully connected layer...', i, ' with ', num_outputs, ' branching is ', branch, 'start branch is : ', str(start_branch)

            if not branch:
                print 'not branch'
                nets[i] = slim.layers.fully_connected(nets[i-1], num_outputs=num_outputs, scope='fc'+str(i),activation_fn=activation, reuse=reuse)
            elif branch:
                print 'branch'
                nets[i] = [None, None, None]
                for j in range(3):
                    if start_branch:
                        print 'start branch..'
                        nets[i][j] = slim.layers.fully_connected(nets[i-1], num_outputs=num_outputs, scope='fc'+str(i)+str(j),activation_fn=activation, reuse=reuse)
                    else:
                        print 'not start branch'
                        nets[i][j] = slim.layers.fully_connected(nets[i-1][j], num_outputs=num_outputs,
                                                                 scope='fc' + str(i)+str(j), activation_fn=activation,
                                                                 reuse=reuse)

        elif layer_type == 'AvgPool2D':
            if not branch:
                nets[i] = slim.layers.avg_pool2d(nets[i-1], [arch['pool_size'], arch['pool_size']])
            elif branch:
                nets[i] = [None, None , None]
                for j in range(3):
                    if start_branch:
                        nets[i][j] = slim.layers.avg_pool2d(nets[i-1], [arch['pool_size'], arch['pool_size']])
                    else:
                        nets[i][j] = slim.layers.avg_pool2d(nets[i-1][j], [arch['pool_size'], arch['pool_size']])

        elif layer_type == 'maxpool2D':
            print 'adding maxpoo2D...', i
            if not branch:
                nets[i] = slim.layers.max_pool2d(nets[i - 1], [arch['pool_size'], arch['pool_size']])
            elif branch:
                nets[i] = [None, None, None]
                for j in range(3):
                    if start_branch:
                        nets[i][j] = slim.layers.max_pool2d(nets[i - 1], [arch['pool_size'], arch['pool_size']])
                    else:
                        nets[i][j] = slim.layers.max_pool2d(nets[i - 1][j], [arch['pool_size'], arch['pool_size']])


        elif layer_type == 'flatten':
            if not branch:
                nets[i] = slim.layers.flatten(nets[i-1], scope='flatten'+str(i))
            elif branch:
                nets[i] = [None, None, None]
                for j in range(3):
                    if start_branch:
                        nets[i][j] = slim.layers.flatten(nets[i-1], scope='flatten'+str(i)+str(j))
                    else:
                        nets[i][j] = slim.layers.flatten(nets[i-1][j], scope='flatten' + str(i)+str(j))
        elif layer_type == 'dropout':
            if not branch:
                nets[i] = tf.nn.dropout(nets[i-1], arch['value'], seed=SEED)
            elif branch:
                nets[i] = [None, None, None]
                for j in range(3):
                    nets[i][j] = tf.nn.dropout(nets[i-1][j], arch['value'], seed=SEED)

    return nets[i]

def error_rate(predictions, labels, step=0, dataset_partition=''):

    predictions = np.swapaxes(predictions, 0, 1)


    l1 = np.mean(np.absolute(predictions[:, 0] - labels[:, 0]))# * (180 / math.pi)
    l2 = np.mean(np.absolute(predictions[:, 1] - labels[:, 1]))# * (180 / math.pi)
    l3 = np.mean(np.absolute(predictions[:, 2] - labels[:, 2]))# * (180 / math.pi)


    t1 = time.clock()
    try:
        dis = compute_disorientations(predictions, labels, is_degree=False)
    except:
        print 'Exception in computing disorientation for ', dataset_partition
        traceback.print_exc(file=sys.stdout)
        dis = -1.0
    return l1, l2, l3, l1+l2+l3, dis

def save_predictions(test_predictions, test_labels, save_dir, test_error1, test_error2, test_error3, test_dis):
    f= open(os.path.join(save_dir, 'test_predictions.txt',), 'w')
    test_predictions = np.swapaxes(test_predictions, 0, 1)
    f.write('Mean disorientation is %.4f, mean errors are %.3f %.3f %.3f\n'%(test_dis, test_error1, test_error2, test_error3))
    f.write('Pred_angle1\tPred_angle2\tPred_angle3\tActual_angle1\tActual_angle2\tActual_angle3\n')
    for i in range(test_predictions.shape[0]):
        f.write('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n'%(test_predictions[i,0], test_predictions[i,1],test_predictions[i,2], test_labels[i,0], test_labels[i,1], test_labels[i,2]))
    f.close()



data_path = 'training-data'
train_file1 = 'EBSDDictionary_100.h5'
train_file2 = 'EBSDDictionary_50.h5'
test_file = 'EBSDrandom.h5'
train_files = [train_file1, train_file2]




def run_model(preprocess=True,parameter='mean',normalizer=True, norm='l1', axis=1, use_valid = True):
    tf.reset_default_graph()

    global data_path, train_file1, train_file2, test_file, train_files
    global batch_size, learning_rate, architecture, num_epochs
    global rr

    Load_Data = LoadData(data_path, train_files, test_file)
    if preprocess is not None:
        if not normalizer:
            rr.fprint('\n\nPreprocessing with parameter: %s norm: %s axis: %d'%(parameter, norm, axis))
        else:
            rr.fprint('\n\nNormalizing with norm: %s axis: %d'%(norm, axis))
        Load_Data.preprocess(preprocess=True,parameter='mean',normalize=normalizer, norm=norm, axis=axis)
    train_X, train_y, valid_X, valid_y, test_X, test_y = Load_Data.get_data(valid=use_valid, target_id=None)

    target_id = 0 ## there are in total 3 dimensions of targets
    train_y = train_y.reshape((len(train_y),3)).astype('float32')
    valid_y = valid_y.reshape((len(valid_y),3)).astype('float32')
    test_y = test_y.reshape((len(test_y),3)).astype('float32')


    train_X = train_X.astype("float32")
    valid_X = valid_X.astype("float32")
    test_X = test_X.astype("float32")
    if not preprocess:
        train_X /= 255
        valid_X /= 255
        test_X /= 255

    train_X = train_X.reshape((-1, 60, 60, 1))
    valid_X = valid_X.reshape((-1, 60, 60, 1))
    test_X = test_X.reshape((-1, 60, 60, 1))

    train_data = train_X
    train_labels = train_y
    test_data = test_X
    test_labels = test_y
    validation_data = valid_X
    validation_labels = valid_y

    rr.fprint("train matrix shape of train_X: ",train_X.shape, ' train_y: ', train_y.shape)
    rr.fprint("valid matrix shape of train_X: ",valid_X.shape, ' valid_y: ', valid_y.shape)
    rr.fprint("test matrix shape of valid_X:  ",test_X.shape, ' test_y: ', test_y.shape)

    train_data_node = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    eval_data = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    logits = model_slim(train_data_node, architecture)
    logits = tf.stack(logits)
    batch = tf.Variable(0)
    train_size = train_X.shape[0]
    learning_rate_ = tf.train.exponential_decay(learning_rate, batch, train_size/batch_size, 0.95, staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)
    train_labels_node1 = tf.placeholder(tf.float32, shape=(batch_size,1))
    train_labels_node2 = tf.placeholder(tf.float32, shape=(batch_size,1))
    train_labels_node3 = tf.placeholder(tf.float32, shape=(batch_size,1))

    loss1 = tf.reduce_mean(tf.abs(train_labels_node1 - logits[0]))  # * (180 / math.pi)
    loss2 = tf.reduce_mean(tf.abs(train_labels_node2 - logits[1]))  # * (180 / math.pi)
    loss3 = tf.reduce_mean(tf.abs(train_labels_node3 - logits[2]))  # * (180 / math.pi)

    loss = loss1 + loss2 + loss3

    actual = tf.stack([train_labels_node1, train_labels_node2, train_labels_node3])
    logits = tf.squeeze(logits)
    actual = tf.squeeze(actual)

    actual = tf.transpose(actual)
    logits = tf.transpose(logits)
    print 'logits: ', logits.get_shape(), logits.dtype
    print 'actual: ', actual.get_shape(), actual.dtype

    print 'building disorientation graph'
    disorients = compute_disorientation_tf(logits, actual)

    disorient = tf.reduce_mean(disorients)
    print 'building optimizer'


    f_loss = disorient + loss
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(f_loss, global_step=batch)

    print 'building evaluation graph'
    eval_prediction = model_slim(eval_data, architecture,train=False)


    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < batch_size:
            raise ValueErro('batch size for evals larger than dataset: %d' % size)
        predictions = np.ndarray(shape=(3, size), dtype=np.float32)
        for begin in xrange(0, size, batch_size):
            end = begin + batch_size
            if end <= size:
                # predictions[:,begin:end] \
                output = sess.run(eval_prediction, feed_dict={eval_data: data[begin:end, ...]})
                # print output
                output = np.squeeze(np.asarray(output))
                predictions[:, begin:end] = output
            else:
                batch_predictions = sess.run(eval_prediction, feed_dict={eval_data: data[-batch_size:, ...]})
                batch_predictions = np.squeeze(np.asarray(batch_predictions))
                predictions[:, -batch_size:] = batch_predictions
        return predictions

    start_time = time.time()
    print 'num_epochs is ', num_epochs
    sess = tf.Session()
    merged = tf.summary.merge_all()
    sess.run(tf.initialize_all_variables())
    rr.fprint('Initialized')


    saver = tf.train.Saver()

    train_writer = tf.summary.FileWriter('summary', graph_def=sess.graph_def)

    best_dis_error = 100
    save_path_ = os.path.join(save_dir, 'model.ckpt')

    for step in xrange(int(num_epochs*train_size) // batch_size +1):
        offset = (step * batch_size) % (train_size - batch_size)

        batch_data = train_data[offset:(offset + batch_size),...]
        batch_labels = train_labels[offset:(offset + batch_size), ...]
        feed_dict = {train_data_node: batch_data,
                     train_labels_node1: np.reshape(batch_labels[:, 0], (batch_size, 1)),
                     train_labels_node2: np.reshape(batch_labels[:, 1], (batch_size, 1)),
                     train_labels_node3: np.reshape(batch_labels[:, 2], (batch_size, 1))}

        _, logits_, l1, l2, l3, l, f_loss_, dis, lr, summ = sess.run([optimizer, logits, loss1, loss2, loss3, loss,f_loss, disorient, learning_rate_, merged], feed_dict=feed_dict)

        if math.isnan(np.sum(logits_)): return

        if step % EVAL_FREQUENCY == 0:
            train_writer.add_summary(summ, step)

            elapsed_time = time.time() - start_time
            if use_valid:
                val_predictions = eval_in_batches(validation_data, sess)
                val_error1, val_error2, val_error3, val_error, val_dis = error_rate(val_predictions, validation_labels, step, 'Validation')
            test_predictions = eval_in_batches(test_data, sess)
            test_error1, test_error2, test_error3, test_error, test_dis = error_rate(test_predictions, test_labels, step,'Test')

            if best_dis_error > test_dis:
                best_dis_error = test_dis
                save_path = saver.save(sess, save_path_)
                rr.fprint('Model saved at: %s' % save_path)
                save_predictions(test_predictions, test_labels, save_dir, test_error1, test_error2, test_error3, test_dis)

            if not use_valid:
                val_error1, val_error2, val_error3, val_error, val_dis = test_error1, test_error2, test_error3, test_error, test_dis

            rr.fprint(
                'Step %d (epoch %.2d), %.1f s f_loss: %f Minibatch loss: %.3f (%.3f, %.3f, %.3f) dis: %.3f, validation loss:  %.3f (%.3f, %.3f, %.3f) dis: %.3f, test loss: %.3f (%.3f, %.3f, %.3f) dis: %.3f learning rate: %.6f' % (
                step, int(step * batch_size) / train_size,
                elapsed_time, f_loss_, l, l1, l2, l3, dis, val_error, val_error1, val_error2, val_error3, val_dis, test_error, test_error1, test_error2, test_error3, test_dis, lr))
            sys.stdout.flush()

            start_time = time.time()

    train_writer.close()
    return

run_model(preprocess=None, use_valid=False)

rr.close()
