__author__ = 'Thushan Ganegedara'

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy as np

from ConvPoolLayer import ConvPoolLayer
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression

import pickle
import os
import math

import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Bool
from rospy_tutorials.msg import Floats
from std_msgs.msg import Int16


def make_shared(batch_x, batch_y, name, normalize, normalize_thresh=1.0):
    '''' Load data into shared variables '''
    print('size x:',batch_x.shape,' size y:',batch_y.shape)
    if not normalize:
        x_shared = theano.shared(batch_x, name + '_x_pkl')
    else:
        x_shared = theano.shared(batch_x, name + '_x_pkl')/normalize_thresh
    max_val = np.max(x_shared.eval())
    print('Max val: ',max_val)
    assert 0.004<=max_val<=1.
    y_shared = T.cast(theano.shared(batch_y.astype(theano.config.floatX), name + '_y_pkl'), 'int32')
    size = batch_x.shape[0]

    return x_shared, y_shared, size

def load_data(filename):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    with open(filename, 'rb') as handle:
        train_set, valid_set, test_set = pickle.load(handle, encoding='latin1')

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def calc_conv_and_pool_params(in_w,in_h,in_ch,fil_w,fil_h, pool_w, pool_h):
    # number of filters per feature map after convolution
    fil_count_w = in_w - fil_w + 1
    fil_count_h = in_h - fil_h + 1

    # number of filters per feature map after max pooling
    out_w = int(fil_count_w/pool_w)
    out_h = int(fil_count_h/pool_h)

    return [out_w,out_h]

def calc_chained_out_shape(layer,img_w,img_h,filters, pools,pooling=True):

    fil_count_w = img_w
    fil_count_h = img_h

    out_w = fil_count_w
    out_h = fil_count_h

    #if i<0
    for i in range(layer+1):
        # number of filters per feature map after convolution
        fil_count_w = out_w - filters[i][0] + 1
        fil_count_h = out_h - filters[i][0] + 1

        # number of filters per feature map after max pooling
        if pooling:
            out_w = int(fil_count_w/pools[i][0])
            out_h = int(fil_count_h/pools[i][0])
        else:
            out_w = fil_count_w
            out_h = fil_count_h

    return out_w,out_h

data_inputs = None
data_labels = None
action_pub = None

img_w = 256 # input image width
img_h = 256 # input image height
out_size = 3
episode=0

def train(batch_size, curr_data, next_data, lr, conv_layers,fulcon_layers,classif_layer):
    epochs = 2

    # create a list of all model parameters to be fit by gradient descent
    params = []
    params += [l.params[0] for l in fulcon_layers] + [l.params[1] for l in fulcon_layers]
    params += [l.params[0] for l in conv_layers] + [l.params[1] for l in conv_layers]
    params += classif_layer.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: curr_data[0][index * batch_size: (index + 1) * batch_size],
            y: curr_data[1][index * batch_size: (index + 1) * batch_size]
        }
    )

    for epoch in range(epochs):
        print("Epoch ",epoch," ...")
        for t_batch in range(int(math.ceil(curr_data[2] / batch_size))):
            train_model(t_batch)

    action = test(batch_size,curr_data[0], conv_layers,fulcon_layers,classif_layer)
    print("Action returned: ", action)
    return action

def test(batch_size,curr_data_x,conv_layers,fulcon_layers,classif_layer):
    last_idx = int(math.ceil(curr_data_x.eval().shape[0]/batch_size))-1
    test_model = theano.function(
        [index],
        classif_layer.y_pred,
        givens={
            x: curr_data_x[index * batch_size: (index + 1) * batch_size],
        }
    )

    action=test_model(last_idx)
    return action[-1]

def run(data_file):
    global episode
    if data_file[1].shape[0]>0:
        shared_data_file = make_shared(data_file[0],data_file[1],'inputs',False)
        print("Running Deep RL Net...")
        print("[run] Input size: ", shared_data_file[0].eval().shape)
        print("[run] Label size: ", shared_data_file[1].eval().shape)
        print("[run] Count: ", shared_data_file[2])
        valid_file = [None,None]
        next_data_file = shared_data_file

        action = -1
        if shared_data_file and shared_data_file[2]>0 and episode<250:
            print("Train Phase ...",episode,"\n")
            action = train(batch_size, shared_data_file, next_data_file, learning_rate,conv_layers,fulcon_layers,classif_layer)
        elif shared_data_file and shared_data_file[2]>0 and episode>=250:
            print("Test Phase ...",episode,"\n")
            action = test(batch_size,shared_data_file[0],conv_layers,fulcon_layers,classif_layer)

        episode = episode+int(shared_data_file[2]/batch_size)
    else:
        action=1


    action_pub.publish(action)


def callback_data_save_status(msg):
    global data_inputs
    global data_labels

    print('Data received')
    run([data_inputs,data_labels])

def callback_data_inputs(msg):
    print('Inputs received')
    global data_inputs
    global img_w,img_h
    data_inputs = np.asarray(msg.data,dtype=np.float32).reshape((img_w*img_h*3,-1))/255.
    data_inputs = data_inputs.T
    print('Input size: ',data_inputs.shape)

def callback_data_labels(msg):
    print('Labels received')
    global data_labels
    global out_size
    data_labels = np.asarray(msg.data,dtype=np.int32).reshape((-1,))
    print('Label size: ',data_labels.shape)

if __name__ == '__main__':
    theano.config.floatX = 'float32'
    rng = np.random.RandomState(23455)

    # kernel size refers to the number of feature maps in a given layer
    # 1st one being number of channels in the image
    conv_activation = 'relu'

    nkerns=[3, 16, 16]
    fulcon_layer_sizes = [512]
    n_conv_layers = len(nkerns)-1
    n_fulcon_layers = len(fulcon_layer_sizes)


    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    batch_size = 2
    learning_rate = 0.1
    pooling = True

    global out_size
    # filter width and height
    filters = [(25,25),(9,9),(3,3)]

    # pool width and height
    pools = [(4,4),(5,5),(2,2)]

    print('Building the model ...')
    print('Pooling: ',pooling,' ',pools)
    print('Learning Rate: ', learning_rate)
    print('Image(Channels x Width x Height): ',nkerns[0],'x',img_w,'x',img_h)

    layer0_input = x.reshape((batch_size,nkerns[0],img_w,img_h))
    in_shapes = [(img_w,img_h)]
    in_shapes.extend([calc_chained_out_shape(i,img_w,img_h,filters,pools,pooling) for i in range(n_conv_layers)])

    print('Convolutional layers')

    conv_layers = [ConvPoolLayer(rng,
                               image_shape=(batch_size,nkerns[i],in_shapes[i][0],in_shapes[i][1]),
                               filter_shape=(nkerns[i+1], nkerns[i], filters[i][0], filters[i][1]),
                               poolsize=(pools[i][0],pools[i][1]),pooling=pooling,activation=conv_activation)
                       for i in range(n_conv_layers)]

    # set the input
    for i,layer in enumerate(conv_layers):
        if i==0:
            input = layer0_input
        else:
            input = conv_layers[i-1].output
        layer.process(input)


    print('\nConvolutional layers created with Max-Pooling ...')

    fulcon_start_in = conv_layers[-1].output.flatten(2)

    fulcon_layers = [
        HiddenLayer(rng,n_in=fulcon_layer_sizes[i-1],n_out=fulcon_layer_sizes[i],activation=T.tanh) if i>0 else
        HiddenLayer(
            rng,
            n_in=nkerns[-1]* in_shapes[-1][0] * in_shapes[-1][1], #if it is maxout there will be only 1 kernel
            n_out=fulcon_layer_sizes[0],activation=T.tanh)
        for i in range(n_fulcon_layers)
    ]

    for i,layer in enumerate(fulcon_layers):
        if i==0:
            input = fulcon_start_in
        else:
            input = fulcon_layers[i-1].output
        layer.process(input)

    print('Fully connected hidden layers created ...')

    classif_layer = LogisticRegression(input=fulcon_layers[-1].output, n_in=fulcon_layer_sizes[-1], n_out=out_size)

    cost = classif_layer.negative_log_likelihood(y)

    rospy.init_node("deep_rl_node")
    action_pub = rospy.Publisher('action_status', Int16, queue_size=10)
    rospy.Subscriber("/data_sent_status", Bool, callback_data_save_status)
    rospy.Subscriber("/data_inputs", numpy_msg(Floats), callback_data_inputs)
    rospy.Subscriber("/data_labels", numpy_msg(Floats), callback_data_labels)

    rospy.spin()
