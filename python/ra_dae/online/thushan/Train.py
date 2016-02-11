

__author__ = 'Thushan Ganegedara'

import pickle
import theano
import theano.tensor as T
import DLModels
import NNLayer
import RLPolicies
import os
import math
import logging
import numpy as np
import time
import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Bool
from rospy_tutorials.msg import Floats
from std_msgs.msg import Int16

def make_shared(batch_x, batch_y, name, normalize, normalize_thresh=1.0):
    '''' Load data into shared variables '''
    if not normalize:
        x_shared = theano.shared(batch_x, name + '_x_pkl')
    else:
        x_shared = theano.shared(batch_x, name + '_x_pkl')/normalize_thresh
    max_val = np.max(x_shared.eval())

    assert 0.004<=max_val<=1.
    y_shared = T.cast(theano.shared(batch_y.astype(theano.config.floatX), name + '_y_pkl'), 'int32')
    size = batch_x.shape[0]

    return x_shared, y_shared, size

def load_from_pickle(filename):

    with open(filename, 'rb') as handle:
        train, valid, test = pickle.load(handle, encoding='latin1')

        train = make_shared(train[0], train[1], 'train', False, 1.0)
        valid = make_shared(valid[0], valid[1], 'valid', False, 1.0)
        test  = make_shared(test[0], test[1], 'test', False, 1.0)

        return train, valid, test

def make_layers(in_size, hid_sizes, out_size, zero_last = False, layer_params = None, init_sizes = None):
    layers = []
    if layer_params is not None:
        W,b,b_prime = layer_params[0]
        print('Restoring (i=0) W: ',W.shape)
        print('Restoring (i=0) b: ',b.shape)
        print('Restoring (i=0) b_prime: ',b_prime.shape)
        print('Restoring (i=0) init_size: ', init_sizes[0])
        layers.append(NNLayer.Layer(in_size, W.shape[1], False, W, b, b_prime,init_sizes[0]))
    else:
        layers.append(NNLayer.Layer(in_size, hid_sizes[0], False, None, None, None))
    for i, size in enumerate(hid_sizes,0):
        if i==0: continue
        if layer_params is not None:
            W,b,b_prime = layer_params[i]
            print('Restoring (i=',i,') W: ',W.shape)
            print('Restoring (i=',i,') b: ',b.shape)
            print('Restoring (i=',i,') b_prime: ',b_prime.shape)
            print('Restoring (i=',i,') init_size: ', init_sizes[i])
            layers.append(NNLayer.Layer(W.shape[0], W.shape[1], False, W, b, b_prime,init_sizes[i]))
        else:
            layers.append(NNLayer.Layer(hid_sizes[i-1], hid_sizes[i], False, None, None, None))

    layers.append(NNLayer.Layer(hid_sizes[-1], out_size, True, None, None, None))
    print('Finished Creating Layers')

    return layers

def make_model(model_type,in_size, hid_sizes, out_size,batch_size, corruption_level, lam, iterations, pool_size, restore_data=None):

    rng = T.shared_randomstreams.RandomStreams(0)
    global episode
    if restore_data is not None:
        layer_params,init_sizes,rl_q_vals,ep = restore_data
        episode = ep
        policy = RLPolicies.ContinuousState(q_vals=rl_q_vals)
        layers = make_layers(in_size, hid_sizes, out_size, False, layer_params, init_sizes)
    else:
        policy = RLPolicies.ContinuousState()
        layers = make_layers(in_size, hid_sizes, out_size, False, None)

    if model_type == 'DeepRL':
        model = DLModels.DeepReinforcementLearningModel(
            layers, corruption_level, rng, iterations, lam, batch_size, pool_size, policy,0.7)
    elif model_type == 'SAE':
        model = DLModels.StackedAutoencoderWithSoftmax(
            layers,corruption_level,rng,lam,iterations)
    elif model_type == 'MergeInc':
        model = DLModels.MergeIncDAE(layers, corruption_level, rng, iterations, lam, batch_size, pool_size)

    model.process(T.matrix('x'), T.ivector('y'))

    return model


def create_image_from_vector(vec, dataset,turn_bw):
    from pylab import imshow,show,cm
    if dataset == 'mnist':
        imshow(np.reshape(vec*255,(-1,28)),cmap=cm.gray)
    elif dataset == 'cifar-10':
        if not turn_bw:
            new_vec = 0.2989 * vec[0:1024] + 0.5870 * vec[1024:2048] + 0.1140 * vec[2048:3072]
        else:
            new_vec = vec
        imshow(np.reshape(new_vec*255,(-1,32)),cmap=cm.gray)
    show()

def train(batch_size, data_file, next_data_file, pre_epochs, fine_epochs, learning_rate, model, modelType):
    t_distribution = []
    start_time = time.clock()

    for arc in range(model.arcs):
        v_errors = []
        results_func = model.error_func

        if modelType == 'DeepRL':
            train_adaptive = model.train_func(arc, learning_rate, data_file[0], data_file[1], next_data_file[0], next_data_file[1], batch_size)
            get_act_vs_pred_train_func = model.act_vs_pred_func(arc, data_file[0], data_file[1], batch_size)
            get_params = model.get_param_values_func()

        elif modelType == 'SAE':
            pretrain_func,finetune_func,finetune_valid_func = model.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size, False, next_data_file[0],next_data_file[1])
            get_act_vs_pred_train_func = model.act_vs_pred_func(arc, data_file[0], data_file[1], batch_size)


        if next_data_file[0] and next_data_file[1]:
            validate_func = results_func(arc, next_data_file[0], next_data_file[1], batch_size)


        print('training data ...')
        try:
            if modelType == 'DeepRL':

                from collections import Counter

                for f_epoch in range(fine_epochs):
                    epoch_start_time = time.clock()
                    print ('\n Fine Epoch: ', f_epoch)
                    fine_tune_costs = []
                    for t_batch in range(int(math.ceil(data_file[2] / batch_size))):
                        train_batch_start_time = time.clock()

                        t_dist = Counter(data_file[1][t_batch * batch_size: (t_batch + 1) * batch_size].eval())
                        t_distribution.append({str(k): v*1.0 / sum(t_dist.values()) for k, v in t_dist.items()})
                        model.set_train_distribution(t_distribution)
                        global episode
                        print('Train batch: ', episode, ' Distribution: ', t_dist)

                        train_adaptive(t_batch,episode)

                        train_batch_stop_time = time.clock()
                        print('\nTime for train batch ', t_batch, ': ', (train_batch_stop_time-train_batch_start_time), ' (secs)')
                        episode=episode+1
                    epoch_stop_time = time.clock()
                    print('Time for epoch ',f_epoch, ': ',(epoch_stop_time-epoch_start_time)/60,' (mins)')
                action = test(data_file[0],arc,model,modelType)
                print("Action returned: ", action)
                return action

        except StopIteration:
            pass

    end_time = time.clock()
    print('\nTime taken for the data stream: ', (end_time-start_time)/60, ' (mins)')
    return

def persist_parameters(layers,policy):
    import pickle
    global episode # number of processed batches

    lyr_params = []
    init_sizes = []
    for l in layers:
        W,b,b_prime = l.get_params()
        init_sizes.append(l.initial_size)
        lyr_params.append((W.get_value(borrow=True),b.get_value(borrow=True),b_prime.get_value(borrow=True)))

    pickle.dump((lyr_params, init_sizes, policy.get_Q(),episode),open( "params.pkl", "wb" ))

def test(shared_data_file_x,arc,model, modelType):
    batch_size =1
    if modelType == 'DeepRL':
        get_action_func = model.get_predictions_func(arc, shared_data_file_x, batch_size)
        last_idx = int(math.ceil(shared_data_file_x.eval().shape[0]/batch_size))-1
        print('Last idx: ',last_idx)
        probs = get_action_func(last_idx)
        print('[test] Probs: ', probs)
        print('[test] Action got: ', np.argmax(probs),'\n')
        return np.argmax(probs)

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
        global train_for
        if shared_data_file and shared_data_file[2]>0 and episode<train_for:
            action = train(batch_size, shared_data_file, next_data_file, pre_epochs, finetune_epochs, learning_rate, model, modelType)
        elif shared_data_file and shared_data_file[2]>0 and episode>=train_for:
            action = test(shared_data_file[0],1,model,modelType)

        global persist_complete
        if episode > 5 and not persist_complete:
            print('Persist parameters')
            persist_parameters(model.layers,model._controller)
            persist_complete = True
    else:
        action=1
    action_pub.publish(action)


def callback_data_save_status(msg):
    global data_inputs
    global data_labels

    print('Running DeepRL ...')
    input_count = data_inputs.shape[0]
    label_count = data_labels.shape[0]
    if data_inputs.shape[0] != data_labels.shape[0]:
        print('data and label counts are different. correcting')
        if label_count >input_count:
            for del_i in range(-1,input_count-label_count-1):
                np.delete(data_labels,del_i,0)
        if label_count < input_count:
            for del_i in range(-1,input_count-label_count-1):
                np.delete(data_inputs,del_i,0)

    run([data_inputs,data_labels])

def callback_data_inputs(msg):
    global data_inputs
    global in_size
    data_inputs = np.asarray(msg.data,dtype=np.float32).reshape((in_size,-1))/255.
    data_inputs = data_inputs.T
    print('Recieved. Input size: ',data_inputs.shape)

def callback_data_labels(msg):
    global data_labels
    global out_size
    data_labels = np.asarray(msg.data,dtype=np.int32).reshape((-1,))
    print('Recieved. Label size: ',data_labels.shape)

data_inputs = None
data_labels = None
action_pub = None

in_size = -1
out_size = 1
episode=0
train_for = 150
restore_last = False
persist_complete = False

if __name__ == '__main__':

    global restore_last
    restore_last = False
    global train_for
    train_for = 150

    import getopt
    import sys
    try:
        opts,args = getopt.getopt(sys.argv[1:],"",["restore_last=","train_for="])
    except getopt.GetoptError:
        print('<filename>.py --restore_last <1 or 0> --train_for <int>')
        sys.exit(2)

    #when I run in command line
    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--restore_last':
                restore_last = bool(arg)
            if opt == '--restore_last':
                train_for = int(arg)


    in_size = 4096
    out_size = 3

    modelType = 'DeepRL'

    learning_rate = 0.2
    batch_size = 5
    epochs = 1
    theano.config.floatX = 'float32'

    hid_sizes = [2048]

    corruption_level = 0.2
    lam = 0.1
    iterations = 10
    pool_size = 50

    pre_epochs = 5
    finetune_epochs = 1

    layers_str = str(in_size) + ', '
    for s in hid_sizes:
        layers_str += str(s) + ', '
    layers_str += str(out_size)

    network_size_logger, reconstruction_err_logger, error_logger = None, None, None

    if restore_last:
        restore_data = pickle.load( open( "params.pkl", "rb" ) )
        model = make_model(modelType,in_size, hid_sizes, out_size, batch_size,corruption_level,lam,iterations,pool_size,restore_data=restore_data)
    else:
        model = make_model(modelType,in_size, hid_sizes, out_size, batch_size,corruption_level,lam,iterations,pool_size)


    model_info = '---------- Model Information -------------\n'
    model_info += 'Model type: ' + modelType + '\n'
    model_info += 'Batch size: ' + str(batch_size) + '\n'
    model_info += 'Epochs: ' + str(epochs) + '\n'

    model_info += 'Network Configuration: ' + layers_str + '\n'
    model_info += 'Iterations: ' + str(iterations) + '\n'
    model_info += 'Lambda Regularizing Coefficient: ' + str(lam) + '\n'
    model_info += 'Pool Size (Train): ' + str(pool_size) + '\n'

    print(model_info)

    rospy.init_node("deep_rl_node")
    action_pub = rospy.Publisher('action_status', Int16, queue_size=10)
    rospy.Subscriber("/data_sent_status", Bool, callback_data_save_status)
    rospy.Subscriber("/data_inputs", numpy_msg(Floats), callback_data_inputs)
    rospy.Subscriber("/data_labels", numpy_msg(Floats), callback_data_labels)

    rospy.spin()


