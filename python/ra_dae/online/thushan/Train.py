

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
from math import ceil

def make_shared(batch_x, batch_y, name, normalize, normalize_thresh=1.0):
    '''' Load data into shared variables '''
    if not normalize:
        x_shared = theano.shared(batch_x, name + '_x_pkl')
    else:
        x_shared = theano.shared(batch_x, name + '_x_pkl')/normalize_thresh
    max_val = np.max(x_shared.eval())

    assert 0.004<=max_val<=1.
    y_shared = T.cast(theano.shared(batch_y.astype(theano.config.floatX), name + '_y_pkl'),'int32')
    size = batch_x.shape[0]

    return x_shared, y_shared, size

def load_from_pickle(filename):

    with open(filename, 'rb') as handle:
        train, valid, test = pickle.load(handle, encoding='latin1')

        train = make_shared(train[0], train[1], 'train', False, 1.0)
        valid = make_shared(valid[0], valid[1], 'valid', False, 1.0)
        test  = make_shared(test[0], test[1], 'test', False, 1.0)

        return train, valid, test

def make_layers(in_size, hid_sizes, out_size, zero_last = False, layer_params = None, init_sizes = None, modelType="DeepRL"):
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

num_classes = 3
def make_model(model_type,in_size, hid_sizes, out_size,batch_size, corruption_level, lam, iterations, pool_size, restore_data=None):

    global num_classes
    rng = T.shared_randomstreams.RandomStreams(0)
    global episode
    if restore_data is not None:
        layer_params,init_sizes,rl_q_vals,ep = restore_data
        episode = ep
        policy = RLPolicies.ContinuousState(q_vals=rl_q_vals)
        layers = make_layers(in_size, hid_sizes, out_size, False, layer_params, init_sizes)
    else:
        policy = RLPolicies.ContinuousState()
        layers = make_layers(in_size, hid_sizes, out_size, False, None,model_type)

    if model_type == 'DeepRL':
        model = DLModels.DeepReinforcementLearningModel(
            layers, corruption_level, rng, iterations, lam, batch_size, pool_size, policy,0.7,num_classes)
    elif model_type == 'SAE':
        model = DLModels.StackedAutoencoderWithSoftmax(
            layers,corruption_level,rng,lam,iterations)
    elif model_type == 'MergeInc':
        model = DLModels.MergeIncDAE(layers, corruption_level, rng, iterations, lam, batch_size, pool_size)

    model.process(T.matrix('x'), T.matrix('y'))

    return model

# KEEP THIS AS 1 otherwise can lead to issues
last_action = 1 # this is important for DeepRLMultiLogreg
i_bumped = False
num_bumps = 0
def train(batch_size, data_file, prev_data_file, pre_epochs, fine_epochs, learning_rate, model, modelType,input_avger):

    start_time = time.clock()

    for arc in range(model.arcs):
        v_errors = []
        results_func = model.error_func

        #avg_inputs = np.empty((data_file[2],in_size),dtype=theano.config.floatX)
        #n_train_b = int(ceil(data_file[2]*1.0 / batch_size))

        #create averaged images from the pool
        #for t_batch_tmp in range(n_train_b):
        #    batch_before_avg = data_file[0].get_value()[t_batch_tmp*batch_size:(t_batch_tmp+1)*batch_size]
        #    batch_avg = input_avger.get_avg_input(theano.shared(batch_before_avg))
        #    avg_inputs[t_batch_tmp*batch_size:(t_batch_tmp+1)*batch_size] = batch_avg
        #th_avg_inputs = theano.shared(avg_inputs,name='avg_inputs')
        #print('[train]avg_in size',avg_inputs.shape)

        if modelType == 'DeepRL':

            #import scipy # use ifyou wanna check images are received correctly
            #scipy.misc.imsave('img'+str(episode)+'.jpg', data_file[0].get_value()[-1,:].reshape(6 4,-1)*255)
            #scipy.misc.imsave('avg_img'+str(episode)+'.jpg', avg_inputs[-1,:].reshape(64,-1)*255)

            y_list = []
            for i,y in enumerate(data_file[1].eval()):
                if y==0:
                    # we don't really use this. because if 0 it means check_fwd is 0
                    y_list.append([0.5,0,0.5])
                if y==1:
                    y_list.append([0,1,0])

            print('[train] y values: ',np.mean(np.asarray(y_list),axis=0))
            train_adaptive = model.train_func(
                arc, learning_rate, data_file[0], theano.shared(np.asarray(y_list,dtype=theano.config.floatX)),
                batch_size)

            check_fwd = model.check_forward(arc, data_file[0], data_file[1], batch_size)

        elif modelType == 'SAE':
            pretrain_func,finetune_func,finetune_valid_func = model.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size, False, None,None)


        print('[train] Training data ...')
        try:
            if modelType == 'DeepRL':

                from collections import Counter
                global last_action,episode,i_bumped,num_bumps

                i_bumped = False
                for t_batch in range(int(ceil(data_file[2]*1.0 / batch_size))):
                    # if we should go forward #no training though
                    # we always train from previous batches
                    if not check_fwd(t_batch):
                        i_bumped = True
                        num_bumps += 1
                        break

                for p_t_batch in range(prev_data_file[0].get_value().shape[0]):
                    t_dist = Counter(prev_data_file[1][p_t_batch * batch_size: (p_t_batch + 1) * batch_size].eval())
                    model.update_train_distribution({str(k): v*1.0 / sum(t_dist.values()) for k, v in t_dist.items()})

                if not i_bumped:

                    print('[train] didnt bump. yay!!! (No training)')
                    '''if last_action == 0:
                        print('[train] I took correct action 0')
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0.9,0.1,0])
                        train_adaptive_prev = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            train_adaptive_prev(p_t_batch)
                    elif last_action == 1:
                        print('[train] I took correct action 1')
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0,1,0])
                        train_adaptive_prev = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            train_adaptive_prev(p_t_batch)
                    elif last_action==2:
                        print('[train] I took correct action 2')
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0,0.1,0.9])
                        train_adaptive_prev = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            train_adaptive_prev(p_t_batch)'''
                    # though we didn't bump we can't say for sure, which direction whould be best
                    # coz we haven't observed other directions, so we give equal probability
                    # for all directions
                    #y_tmp = []
                    #for i in range(prev_data_file[0].get_value().shape[0]):
                    #    y_tmp.append([0.33,0.34,0.33])
                    #train_adaptive_prev = model.train_func(
                    #    arc, learning_rate, prev_data_file[0],
                    #    theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX)), batch_size)
                    #for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                    #    train_adaptive_prev(p_t_batch)

                # don't update last_action here, do it in the test
                # we've bumped
                if i_bumped:
                    print('[train] bumped after taking action ', last_action )
                    #p_for_batch = get_proba_func(t_batch)
                    #act_for_batch = np.argmax(p_for_batch,axis=0)
                    if last_action == 0:
                        # train using [0,0.5,0.5]
                        print('[train] I shouldve taken action 1 or 2')
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0,0.5,0.5])
                        train_adaptive_prev = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            train_adaptive_prev(p_t_batch)

                    elif last_action == 1:
                        # train adaptively using [0.5, 0, 0.5]
                        print('[train] I shouldve taken action 0 or 2')
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0.5,0.0,0.5])
                        train_adaptive_prev = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            train_adaptive_prev(p_t_batch)
                        # no point in takeing actions here, coz we've bumped

                    else:
                        print('[train] I shouldve taken action 0 or 1')
                        # train_using [0.5,0.5,0]
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0.5,0.5,0])
                        train_adaptive_prev = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            train_adaptive_prev(p_t_batch)

        except StopIteration:
            pass

    end_time = time.clock()
    print('\n[train] Time taken for the episode: ', (end_time-start_time)/60, ' (mins)')
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
    global i_bumped
    if modelType == 'DeepRL':
        get_proba_func = model.get_predictions_func(arc, shared_data_file_x, batch_size)
        last_idx = int(ceil(shared_data_file_x.eval().shape[0]*1.0/batch_size))-1
        print('[test] Last idx: ',last_idx)
        probs = np.mean(get_proba_func(last_idx),axis=0)
        print('[test] Probs: ', probs)
        if np.max(probs)>0.34:
            action = np.argmax(probs)
        else:
            action = np.random.randint(0,3)
        print('[test] Action got: ', action)
        return action

fwd_threshold = 0.5
def run(data_file,prev_data_file):
    global episode,i_bumped,bump_episode,last_action,fwd_threshold,num_bumps
    print('[run] episode: ',episode)
    print('[run] bump_episode: ',bump_episode)
    print('[run] number of bumps: ',num_bumps)
    # this part is for the very first action after bumping somewhere
    if data_file[1].shape[0]>0 and episode-1 == bump_episode:
        print('[run] very first episode after bump')
        last_action = 1
        shared_data_file = make_shared(data_file[0],data_file[1],'inputs',False)
        action = test(shared_data_file[0],1,model,modelType)
    elif data_file[1].shape[0]>0:
        shared_data_file = make_shared(data_file[0],data_file[1],'inputs',False)
        print("Running Deep RL Net...")
        print("[run] Input size: ", shared_data_file[0].eval().shape)
        print("[run] Label size: ", shared_data_file[1].eval().shape)
        print("[run] Count: ", shared_data_file[2])

        if prev_data_file is not None:
            prev_shared_data_file = make_shared(prev_data_file[0],prev_data_file[1],'prev_inputs',False)
        else:
            prev_shared_data_file = None

        action = -1
        global train_for
        if shared_data_file and shared_data_file[2]>0 and num_bumps<train_for:
            train(batch_size, shared_data_file, prev_shared_data_file,
                           pre_epochs, finetune_epochs, learning_rate, model, modelType, input_avger)
            action = test(shared_data_file[0],1,model,modelType)

        elif shared_data_file and shared_data_file[2]>0 and num_bumps>=train_for:
            action = test(shared_data_file[0],1,model,modelType)

        global persist_complete
        global train_for
        if num_bumps >= train_for and not persist_complete:
            print('[run] Persist parameters')
            persist_parameters(model.layers,model._controller)
            persist_complete = True
    else:
        action=1

    last_action = action
    print('[run] last action: ', last_action)
    print "\n"

    episode += 1
    action_pub.publish(action)

prev_data = None
bump_episode = -1
def callback_data_save_status(msg):
    global data_inputs,data_labels,prev_data,i_bumped,bump_episode,episode

    print('[callback] Running DeepRL ...')
    input_count = data_inputs.shape[0]
    label_count = data_labels.shape[0]
    print('[callback] currdata (before): ',data_inputs.shape,', ',data_labels.shape)
    if data_inputs.shape[0] != data_labels.shape[0]:
        print('[callback] data and label counts are different. correcting')
        if label_count >input_count:
            for _ in range(label_count-input_count):
                data_labels = np.delete(data_labels,-1,0)
        if label_count < input_count:
            for _ in range(input_count-label_count):
                data_inputs = np.delete(data_inputs,-1,0)

    # for the 1st iteration
    if i_bumped:
        prev_data = None
        bump_episode = episode-1
        i_bumped = False

    if prev_data is not None:
        print('[callback] prevdata: ',prev_data[0].shape,' ,',prev_data[1].shape)

    print('[callback] currdata (after): ',data_inputs.shape,' ,',data_labels.shape)

    if data_inputs.shape[0]>0 and data_labels.shape[0]>0:
        run([data_inputs,data_labels],prev_data)
        prev_data = [data_inputs,data_labels]
    else:
        print("[callback] No data to run")



def callback_data_inputs(msg):
    global data_inputs
    global in_size
    data_inputs = np.asarray(msg.data,dtype=np.float32).reshape((-1,in_size))/255.
    data_inputs = data_inputs
    #import scipy # use if you wanna check algo receive images correctly
    #scipy.misc.imsave('rec_img'+str(episode)+'.jpg', data_inputs[-1].reshape(64,-1)*255)
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
    global train_for


    import getopt
    import sys
    from input_averager import InputAverager

    try:
        opts,args = getopt.getopt(sys.argv[1:],"",["restore_last=","train_for="])
    except getopt.GetoptError:
        print('<filename>.py --restore_last=<1 or 0> --train_for=<int>')
        sys.exit(2)

    #when I run in command line
    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--restore_last':
                print('restore_last: ',arg)
                restore_last = bool(int(arg))
                print('bool: ',restore_last)
            if opt == '--train_for':
                print('train_for: ',arg)
                train_for = int(arg)


    in_size = 4096
    out_size = 3

    modelType = 'DeepRL'

    learning_rate = 0.2
    batch_size = 5
    epochs = 1
    theano.config.floatX = 'float32'

    hid_sizes = [64]

    corruption_level = 0.2
    lam = 0.1
    iterations = 10
    pool_size = 50

    pre_epochs = 5
    finetune_epochs = 1

    last_batches_pool_size = 5
    last_batches_pool = []

    layers_str = str(in_size) + ', '
    for s in hid_sizes:
        layers_str += str(s) + ', '
    layers_str += str(out_size)

    network_size_logger, reconstruction_err_logger, error_logger = None, None, None

    if restore_last:
        restore_data = pickle.load( open( "params.pkl", "rb" ) )
        model = make_model(modelType,in_size, hid_sizes, out_size, batch_size,corruption_level,lam,iterations,pool_size,restore_data=restore_data)
    else:
        model = make_model(modelType,in_size, hid_sizes, out_size, batch_size,corruption_level,lam,iterations,pool_size,restore_data=None)

    batch_count = 5
    input_avger = InputAverager(batch_count,batch_size,in_size)

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


