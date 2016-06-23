

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
from std_msgs.msg import Bool,Int16
from rospy_tutorials.msg import Floats
from std_msgs.msg import Int16
from math import ceil
import logging
import sys
from threading import Lock
import random


logging_level = logging.DEBUG

logging_format = '[%(name)s] [%(funcName)s] %(message)s'
bump_logging_format = '%(message)s'

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

def make_layers(hparams, layer_params = None, init_sizes = None):
    layers = []
    global logger
    if layer_params is not None:
        logger.info('RESTORING LAYERS FROM PAST DATA ...\n')
        logger.info("Layers: %s \n",len(layer_params))
    else:
        logger.info('CREATING LAYERS ...\n')
    # layer 0
    if layer_params is not None:
        W,b,b_prime = layer_params[0]
        logger.info('Restoring (i=0) W: %s',W.shape)
        logger.info('Restoring (i=0) b: %s',b.shape)
        logger.info('Restoring (i=0) b_prime: %s',b_prime.shape)
        logger.info('Restoring (i=0) init_size: %s \n', init_sizes[0])
        layers.append(NNLayer.Layer(hparams.in_size, W.shape[1], False, W, b, b_prime,init_sizes[0]))
    else:
        nn_layer = NNLayer.Layer(hparams.in_size, hparams.hid_sizes[0], False, None, None, None,None)
        layers.append(nn_layer)

        logger.info('Creating (i=0) W: %s',nn_layer.W.get_value().shape)
        logger.info('Creating (i=0) b: %s',nn_layer.b.get_value().shape)
        logger.info('Creating (i=0) b_prime: %s',nn_layer.b_prime.get_value().shape)
        logger.info('Creating (i=0) init_size: %s \n', nn_layer.initial_size)

    # middle layeres
    for i, size in enumerate(hparams.hid_sizes,0):
        if i==0: continue
        if layer_params is not None:
            W,b,b_prime = layer_params[i]
            logger.info('Restoring (i=',i,') W: %s',W.shape)
            logger.info('Restoring (i=',i,') b: %s',b.shape)
            logger.info('Restoring (i=',i,') b_prime: %s',b_prime.shape)
            logger.info('Restoring (i=',i,') init_size: %s \n', init_sizes[i])
            layers.append(NNLayer.Layer(W.shape[0], W.shape[1], False, W, b, b_prime,init_sizes[i]))
        else:
            nn_layer = NNLayer.Layer(hparams.hid_sizes[i-1], hparams.hid_sizes[i], False, None, None, None,None)
            layers.append(nn_layer)
            logger.info('Creating (i=',i,') W: %s',nn_layer.W.get_value().shape)
            logger.info('Creating (i=',i,') b: %s',nn_layer.b.get_value().shape)
            logger.info('Creating (i=',i,') b_prime: %s',nn_layer.b_prime.get_value().shape)
            logger.info('Creating (i=',i,') init_size: %s\n', nn_layer.initial_size)

    if layer_params is not None:
        if hparams.model_type=='DeepRL' or hparams.model_type=='SDAE':
            W,b,b_prime = layer_params[-1]
            logger.info('Restoring (i=-1) W: %s',W.shape)
            logger.info('Restoring (i=-1) b: %s',b.shape)
            logger.info('Restoring (i=-1) b_prime: %s',b_prime.shape)
            logger.info('Restoring (i=-1) init_size: %s\n', init_sizes[-1])

            layers.append(NNLayer.Layer(hparams.hid_sizes[-1], hparams.out_size, False, W, b, b_prime,init_sizes[-1]))

        elif hparams.model_type=='DeepRLMultiSoftmax':
            out_layers = []
            W_str,b_str,b_prime_str,init_size_str = '','','',''
            for init,softmax in zip(init_sizes[-1],layer_params[-1]):
                W,b,b_prime = softmax
                W_str += str(W.shape)+ ' '
                b_str += str(b.shape)+ ' '
                b_prime_str += str(b_prime.shape)+ ' '
                init_size_str += str(init) + ' '

                logger.info('Restoring (i=-1)(',len(out_layers),') W: %s',W_str)
                logger.info('Restoring (i=-1)(',len(out_layers),') b: %s',b_str)
                logger.info('Restoring (i=-1)(',len(out_layers),') b_prime: %s',b_prime_str)
                logger.info('Restoring (i=-1)(',len(out_layers),') init_size: %s\n', init_size_str)

                out_layers.append(NNLayer.Layer(hparams.hid_sizes[-1], 1, False, W, b, b_prime,init))

            layers.append(out_layers)

    else:
        if hparams.model_type=='DeepRL' or hparams.model_type=='SDAE':
            nn_layer = NNLayer.Layer(hparams.hid_sizes[-1], hparams.out_size, False, None, None, None,None)
            logger.info('Creating (i=-1) W: %s',nn_layer.W.get_value().shape)
            logger.info('Creating (i=-1) b: %s',nn_layer.b.get_value().shape)
            logger.info('Creating (i=-1) b_prime: %s',nn_layer.b_prime.get_value().shape)
            logger.info('Creating (i=-1) init_size: %s \n', nn_layer.initial_size)

            layers.append(nn_layer)
        elif hparams.model_type == 'DeepRLMultiSoftmax':
            out_layers = []
            W_str,b_str,b_prime_str,init_size_str = '','','',''
            for o in range(hparams.out_size):
                nn_layer = NNLayer.Layer(hparams.hid_sizes[-1], 1, False, None, None, None,None)
                W_str += str(nn_layer.W.get_value().shape) + ' '
                b_str += str(nn_layer.W.get_value().shape) + ' '
                b_prime_str += str(nn_layer.b_prime.get_value().shape) + ' '
                init_size_str += str(nn_layer.initial_size) + ' '
                out_layers.append(nn_layer)

            logger.info('Creating (i=-1)(%s) W: %s',len(out_layers),W_str)
            logger.info('Creating (i=-1)(%s) b: %s',len(out_layers),b_str)
            logger.info('Creating (i=-1)(%s) b_prime: %s',len(out_layers),b_prime_str)
            logger.info('Creating (i=-1)(%s) init_size: %s \n',len(out_layers), init_size_str)
            layers.append(out_layers)

    logger.info('LAYER CREATION FINISHED ...\n')

    return layers


def make_model(hparams, restore_data=None,restore_pool=None):
    # restoredata should have layer
    rng = T.shared_randomstreams.RandomStreams(0)
    deeprl_ep = 0
    global episode,num_bumps,deeprl_episodes,algo_move_count
    global logger

    # if initially starting (not loading previous data
    if restore_data is not None:
        layer_params,init_sizes,rl_q_vals,(ep,nb,deeprl_ep,algo_moves) = restore_data
        episode = ep
        num_bumps = nb
        algo_move_count = algo_moves

        if hparams.model_type == 'DeepRL':
            policies = [RLPolicies.ContinuousState(q_vals=q) for q in rl_q_vals]
            assert len(policies) == len(hparams.hid_sizes)
            logger.debug('All policies are correctly restored for DeepRL\n')
            layers = make_layers(hparams, layer_params, init_sizes)

        elif hparams.model_type == 'DeepRLMultiSoftmax':
            policies = [[RLPolicies.ContinuousState(q_vals=q) for q in rl_q_1] for rl_q_1 in rl_q_vals]
            # check if all policies are loaded
            assert len(policies)*len(policies[0]) == hparams.out_size*len(hparams.hid_sizes)
            logger.debug('All policies are correctly restored for DeepRLMultiSoftmax\n')
            layers = make_layers(hparams, layer_params, init_sizes)

        elif hparams.model_type == 'LogisticRegression':
            W,b,b_prime = layer_params[0]
            layers = [NNLayer.Layer(hparams.in_size, hparams.out_size, False, W, b, b_prime,init_sizes[0])]

        elif hparams.model_type == 'SDAE':
            layers = make_layers(hparams,layer_params,init_sizes)

    else:
        if hparams.model_type == 'DeepRL':
            policies = [RLPolicies.ContinuousState() for _ in range(len(hparams.hid_sizes))]
            assert len(policies) == len(hparams.hid_sizes)
            logger.debug('All policies are correctly created for DeepRL\n')
            layers = make_layers(hparams)

        elif hparams.model_type == 'DeepRLMultiSoftmax':
            policies = [[RLPolicies.ContinuousState() for _ in range(len(hparams.hid_sizes))] for __ in range(hparams.out_size)]
            assert len(policies)*len(policies[0]) == hparams.out_size*len(hparams.hid_sizes)
            logger.debug('All policies are correctly created for DeepRLMultiSoftmax\n')
            layers = make_layers(hparams)

        elif hparams.model_type == 'LogisticRegression':
            layers = [NNLayer.Layer(hparams.in_size, hparams.out_size, False, None, None, None,None)]

        elif hparams.model_type == 'SDAE':
            layers = make_layers(hparams)

    if hparams.model_type == 'DeepRL':
        model = DLModels.DeepReinforcementLearningModel(layers, rng, policies, hparams, hparams.out_size)
        model.set_research_params(pool_with_not_bump=pool_with_not_bump,test_mode=False)

        if restore_pool is not None:
            pool,diff_pool = restore_pool
            p_data = theano.shared(np.asarray(pool[0], dtype=theano.config.floatX), 'pool' )
            p_data_y = theano.shared(np.asarray(pool[1], dtype=theano.config.floatX), 'pool_y')

            dp_data = theano.shared(np.asarray(diff_pool[0], dtype=theano.config.floatX), 'diff_pool' )
            dp_data_y = theano.shared(np.asarray(diff_pool[1], dtype=theano.config.floatX), 'diff_pool_y')

            model.restore_pool(hparams.batch_size,p_data,p_data_y,dp_data,dp_data_y)
            model.set_episode_count(deeprl_ep)

    elif hparams.model_type == 'DeepRLMultiSoftmax':
        logger.info("CREATING DEEPRL MULTI SOFTMAX NET ...\n")
        model = DLModels.DeepReinforcementLearningModelMultiSoftmax(layers, rng, policies, hparams, hparams.out_size)

        if restore_pool is not None:
            X,Y,DX,DY = [],[],[],[]

            for pools in restore_pool:
                pool,diff_pool = pools
                X.append(theano.shared(np.asarray(pool[0], dtype=theano.config.floatX), 'pool' ))
                Y.append(theano.shared(np.asarray(pool[1], dtype=theano.config.floatX), 'pool_y'))

                DX.append(theano.shared(np.asarray(diff_pool[0], dtype=theano.config.floatX), 'diff_pool' ))
                DY.append(theano.shared(np.asarray(diff_pool[1], dtype=theano.config.floatX), 'diff_pool_y'))

            model.restore_pool(hparams.batch_size,X,Y,DX,DY)
            model.set_episode_count(deeprl_ep)

    elif hparams.model_type == 'SDAE':
        logger.info('CREATING SDAE ...\n')
        model = DLModels.StackedAutoencoderWithSoftmax(layers,rng,hparams)

    elif hparams.model_type == 'LogisticRegression':
        logger.info('CREATING LOGISTIC REGRESSION ...\n')
        model = DLModels.LogisticRegression(layers[0],hparams)

    return model

# KEEP THIS AS 1 otherwise can lead to issues
last_action = 1 # this is important for DeepRLMultiLogreg
i_bumped = False
num_bumps = 0


def train_sdae(batch_size, data_file, prev_data_file, learning_rate, model, model_type):
    global logger,logging_level,logging_format
    global last_action,num_bumps,i_bumped
    global time_logger,episode

    model.process(training=False)


    check_fwd = model.check_forward(data_file[0], data_file[1], batch_size)

    i_bumped = False

    for t_batch in range(int(ceil(data_file[2]*1.0 / batch_size))):
        if not check_fwd(t_batch):
            i_bumped = True
            num_bumps += 1
            break

    if not i_bumped:
        logger.info('Didnt bump. yay!!!')
        logger.debug('I took correct action %s',last_action)

        y_tmp = []
        for i in range(prev_data_file[0].get_value().shape[0]):
            y_tmp.append(last_action)

        shared_y = theano.shared(np.asarray(y_tmp))
        pre_train_func,finetune_func = model.train_func(
            prev_data_file[0], T.cast(shared_y,'int32'))
        start_time = time.clock()
        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
            pre_train_func(p_t_batch)
            finetune_func(p_t_batch)
        end_time = time.clock()

    if i_bumped:

        logger.info('Bumped after taking action %s', last_action)

        y_tmp = []
        pos_actions =  [0,1,2]
        pos_actions.remove(int(last_action))
        for i in range(prev_data_file[0].get_value().shape[0]):
            y_tmp.append(random.choice(pos_actions))

        shared_y = theano.shared(np.asarray(y_tmp))

        pre_train_func,finetune_func = model.train_func(
            prev_data_file[0],T.cast(shared_y,'int32'),learning_rate=learning_rate/2.0)

        start_time = time.clock()
        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
            pre_train_func(p_t_batch)
            finetune_func(p_t_batch)

        end_time = time.clock()

    logger.info('Time taken for the episode: %10.3f (mins)', (end_time-start_time)/60)
    time_logger.info('%s, %.3f',episode,(end_time-start_time))

    return


def train_logistic_regression(batch_size, data_file, prev_data_file, learning_rate, model, model_type):
    global logger,logging_level,logging_format
    global last_action,num_bumps,i_bumped
    global time_logger,episode

    model.process()



    check_fwd = model.check_forward(data_file[0], data_file[1], batch_size)

    i_bumped = False

    for t_batch in range(int(ceil(data_file[2]*1.0 / batch_size))):
        if not check_fwd(t_batch):
            i_bumped = True
            num_bumps += 1
            break

    if not i_bumped:
        logger.info('Didnt bump. yay!!!')
        logger.debug('I took correct action %s',last_action)

        y_tmp = []
        for i in range(prev_data_file[0].get_value().shape[0]):
            y_tmp.append(last_action)

        shared_y = theano.shared(np.asarray(y_tmp))
        train = model.train_func(
            prev_data_file[0], T.cast(shared_y,'int32'))
        start_time = time.clock()
        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
            train(p_t_batch)
        end_time = time.clock()

    if i_bumped:

        logger.info('Bumped after taking action %s', last_action)

        y_tmp = []
        pos_actions =  [0,1,2]
        pos_actions.remove(int(last_action))
        for i in range(prev_data_file[0].get_value().shape[0]):
            y_tmp.append(random.choice(pos_actions))

        shared_y = theano.shared(np.asarray(y_tmp))

        train = model.train_func(
            prev_data_file[0],T.cast(shared_y,'int32'),learning_rate=learning_rate/2.0)

        start_time = time.clock()
        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
            train(p_t_batch)
        end_time = time.clock()

    logger.info('Time taken for the episode: %10.3f (mins)', (end_time-start_time)/60)
    time_logger.info('%s, %.3f',episode,(end_time-start_time))
    return


def train_multi_softmax(batch_size, data_file, prev_data_file, pre_epochs, fine_epochs, learning_rate, model, model_type):
    global episode
    global logger,logging_level,logging_format
    global time_logger
    model.process(T.matrix('x'), T.matrix('y'),training=True)

    arc = 0
    if model_type == 'DeepRLMultiSoftmax':

        #import scipy # use ifyou wanna check images are received correctly
        #scipy.misc.imsave('img'+str(episode)+'.jpg', data_file[0].get_value()[-1,:].reshape(6 4,-1)*255)
        #scipy.misc.imsave('avg_img'+str(episode)+'.jpg', avg_inputs[-1,:].reshape(64,-1)*255)
        check_fwd = model.check_forward(arc, data_file[0], data_file[1], batch_size)

    logger.info('\nTRANING DATA ...\n')
    try:
        if model_type == 'DeepRLMultiSoftmax':

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

            if not i_bumped:

                for p_t_batch in range(prev_data_file[0].get_value().shape[0]):
                    t_dist = Counter(prev_data_file[1][p_t_batch * batch_size: (p_t_batch + 1) * batch_size].eval())
                    model.update_train_distribution({str(k): v*1.0 / sum(t_dist.values()) for k, v in t_dist.items()},last_action)

                logger.info('Didnt bump. yay!!!')
                logger.debug('I took correct action %s',last_action)

                y_tmp = []
                for i in range(prev_data_file[0].get_value().shape[0]):
                    y_tmp.append(1.)
                train_adaptive_prev,update_pool = model.train_func(
                    arc, learning_rate, prev_data_file[0],
                    theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX).reshape(-1,1)), batch_size,last_action)
                #for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):

                start_time = time.clock()
                for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size)/2),int(ceil(prev_data_file[2]*1.0/batch_size))):
                    train_adaptive_prev(p_t_batch)
                    #update_pool(p_t_batch) # dont do this

                    model.rectify_multi_softmax_layers(last_action)
                end_time = time.clock()

            # don't update last_action here, do it in the test
            # we've bumped
            if i_bumped:

                logger.info('Bumped after taking action %s', last_action)

                for p_t_batch in range(prev_data_file[0].get_value().shape[0]):
                    t_dist = Counter(prev_data_file[1][p_t_batch * batch_size: (p_t_batch + 1) * batch_size].eval())
                    model.update_train_distribution({str(k): v*1.0 / sum(t_dist.values()) for k, v in t_dist.items()},last_action)

                y_tmp = []
                for i in range(prev_data_file[0].get_value().shape[0]):
                    y_tmp.append(0.)

                y = np.asarray(y_tmp).reshape(-1,1)

                train_adaptive_prev,_ = model.train_func(
                    arc, learning_rate, prev_data_file[0],
                    theano.shared(np.asarray(y,dtype=theano.config.floatX)), batch_size,last_action)
                #for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                start_time = time.clock()
                for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size)/2),int(ceil(prev_data_file[2]*1.0/batch_size))):
                    train_adaptive_prev(p_t_batch)
                    model.rectify_multi_softmax_layers(last_action)
                end_time = time.clock()

    except StopIteration:
        pass


    logger.info('Time taken for the episode: %10.3f (secs)', (end_time-start_time))
    time_logger.info('%s, %.3f',episode,(end_time-start_time))
    return


def persist_parameters(updated_hparam,layers,policies,pools,deeprl_episodes):
    import pickle
    global episode, num_bumps, algo_move_count # number of processed batches
    global dir_suffix,logger

    lyr_params = []
    layer_sizes = []
    policy_Qs = []

    for l in layers[:-1]:
        W_tensor,b_tensor,b_prime_tensor = l.get_params()
        W,b,b_prime = W_tensor.get_value(borrow=True),b_tensor.get_value(borrow=True),b_prime_tensor.get_value(borrow=True)

        layer_sizes.append(W.shape)
        lyr_params.append((W,b,b_prime))
        logger.debug('W type: %s',type(W))
        logger.debug('b type: %s',type(b))
        logger.debug('b_prime type: %s',type(b_prime))
        assert isinstance(W,np.ndarray) and isinstance(b,np.ndarray) and isinstance(b_prime,np.ndarray)
        logger.debug('W,b,b_prime added as numpy arrays')

    if updated_hparam.model_type == 'DeepRL':
        W_tensor,b_tensor,b_prime_tensor = layers[-1].get_params()
        W,b,b_prime = W_tensor.get_value(borrow=True),b_tensor.get_value(borrow=True),b_prime_tensor.get_value(borrow=True)

        layer_sizes.append(W.shape)
        lyr_params.append((W,b,b_prime))

        assert isinstance(W,np.ndarray) and isinstance(b,np.ndarray) and isinstance(b_prime,np.ndarray)
        logger.debug('W,b,b_prime added as numpy arrays')

        for p in policies:
            policy_Qs.append(p.get_Q())

    elif updated_hparam.model_type == 'DeepRLMultiSoftmax':
        multi_softmax_layer_params = []
        softmax_layer_sizes = []
        for softmax in layers[-1]:
            W_soft_tensor, b_soft_tensor, b_prime_soft_tensor = softmax.get_params()
            W_soft,b_soft,b_prime_soft = W_soft_tensor.get_value(borrow=True),b_soft_tensor.get_value(borrow=True),b_prime_soft_tensor.get_value(borrow=True)
            multi_softmax_layer_params.append([W_soft,b_soft,b_prime_soft])
            softmax_layer_sizes.append(W_soft.shape)

            logger.debug('W type_soft: %s',type(W_soft))
            logger.debug('b type_soft: %s',type(b_soft))
            logger.debug('b_prime_soft type: %s',type(b_prime_soft))
            assert isinstance(W_soft,np.ndarray) and isinstance(b_soft,np.ndarray) and isinstance(b_prime_soft,np.ndarray)
            logger.debug('W,b,b_prime (softmax) added as numpy arrays')

        lyr_params.append(multi_softmax_layer_params)
        layer_sizes.append(softmax_layer_sizes)

        for drl_pol in policies:
            p_all = []
            for p in drl_pol:
                p_all.append(p.get_Q())
            policy_Qs.append(p_all)

    elif updated_hparam.model_type == 'SDAE':
        W_tensor,b_tensor,b_prime_tensor = layers[-1].get_params()
        W,b,b_prime = W_tensor.get_value(borrow=True),b_tensor.get_value(borrow=True),b_prime_tensor.get_value(borrow=True)

        layer_sizes.append(W.shape)
        lyr_params.append((W,b,b_prime))

        assert isinstance(W,np.ndarray) and isinstance(b,np.ndarray) and isinstance(b_prime,np.ndarray)
        logger.debug('W,b,b_prime added as numpy arrays')

    #TODO: Logistic Regression and SDAE peristance of parameters

    if updated_hparam.model_type == 'DeepRL' or updated_hparam.model_type == 'DeepRLMultiSoftmax':
        assert len(lyr_params)>0 and len(policy_Qs)>0 and len(layer_sizes)>0

    file_suffix = 'in'+ str(updated_hparam.in_size) + '_out' + str(updated_hparam.out_size)\
                  + '_type' + updated_hparam.model_type + '_act-'+ hyperparam.activation \
                  + '_dropout' + str(hyperparam.dropout) + '_lr' + str(updated_hparam.learning_rate)\
                  + '_batch' + str(updated_hparam.batch_size) + '_hid' + '-'.join([str(h) for h in updated_hparam.init_sizes])\
                  + '_sim' + str(updated_hparam.sim_thresh)


    pickle.dump((updated_hparam,lyr_params, layer_sizes, policy_Qs,(episode,num_bumps,deeprl_episodes,algo_move_count)),
                open(dir_suffix + os.sep + "params_"+str(algo_move_count)+ file_suffix + ".pkl", "wb"))

    if updated_hparam.model_type=='DeepRL' or updated_hparam.model_type=='DeepRLMultiSoftmax':
        pickle.dump(pools,open(dir_suffix + os.sep + 'pools_'+str(algo_move_count)+ file_suffix + '.pkl', 'wb'))
    
def test(shared_data_file_x,arc,model, model_type):

    global episode,i_bumped,hyperparam
    global logger,test_time_logger

    if model_type == 'DeepRLMultiSoftmax':
        model.process(T.matrix('x'), T.matrix('y'),training=False)
    elif model_type == 'LogisticRegression':
        model.process()
    elif model_type == 'SDAE':
        model.process(training=False)

    if model_type == 'DeepRL':
        get_proba_func = model.get_predictions_func(arc, shared_data_file_x, hyperparam.batch_size)
        last_idx = int(ceil(shared_data_file_x.eval().shape[0]*1.0/hyperparam.batch_size))-1
        logger.debug('Last idx: %s',last_idx)
        start_time = time.clock()
        probs = np.mean(get_proba_func(last_idx),axis=0)
        end_time = time.clock()

    elif model_type == 'DeepRLMultiSoftmax':
        get_proba_func = model.get_predictions_func(arc, shared_data_file_x, hyperparam.batch_size)
        last_idx = int(ceil(shared_data_file_x.eval().shape[0]*1.0/hyperparam.batch_size))-1
        logger.debug('Last idx: %s',last_idx)
        probs = None
        start_time = time.clock()
        for func in get_proba_func:
            if probs is None:
                probs = np.mean(func(last_idx),axis=0).reshape(-1,1)
            else:
                probs = np.append(probs,np.mean(func(last_idx),axis=0).reshape(-1,1),axis=1)

        probs = probs[0]
        end_time = time.clock()

    elif model_type == 'LogisticRegression':
        get_proba_func = model.get_predictions_func(shared_data_file_x, hyperparam.batch_size)
        last_idx = int(ceil(shared_data_file_x.eval().shape[0]*1.0/hyperparam.batch_size))-1
        logger.debug('Last idx: %s',last_idx)

        start_time = time.clock()
        probs = np.mean(get_proba_func(last_idx),axis=0)
        end_time = time.clock()

    elif model_type == 'SDAE':
        get_proba_func = model.get_predictions_func(arc,shared_data_file_x, hyperparam.batch_size)
        last_idx = int(ceil(shared_data_file_x.eval().shape[0]*1.0/hyperparam.batch_size))-1
        logger.debug('Last idx: %s',last_idx)

        start_time = time.clock()
        probs = np.mean(get_proba_func(last_idx),axis=0)
        end_time = time.clock()

    logger.info('Probs: %s', probs)
    test_time_logger.info('%s, %.3f',episode,(end_time-start_time))

    if model_type == 'DeepRL':
        random_threshold = 0.35
        if np.max(probs)>random_threshold:
            action = np.argmax(probs)
        else:
            action = np.random.randint(0,3)

    if model_type == 'LogisticRegression' or model_type== 'SDAE':
        random_threshold = 0.25
        if np.max(probs)>random_threshold:
            action = np.argmax(probs)
        else:
            action = np.random.randint(0,3)

    elif model_type == 'DeepRLMultiSoftmax':
        #random_threshold = 0.9
        rand_threshold = 0.5 * (1.-hyperparam.dropout) * 0.9
        idx_above_thresh = np.where(probs>rand_threshold)[0]
        is_all_below_low_thresh = np.all(probs<rand_threshold)
        logger.debug('Indices above random threshold: %s',idx_above_thresh)
        logger.debug('All probas below low rand threshold: %s',is_all_below_low_thresh)
        if len(idx_above_thresh)<=1 and not is_all_below_low_thresh:
            action = np.argmax(probs)
            logger.info('Action got: %s \n', action)
        else:

            action = random.choice(idx_above_thresh) if len(idx_above_thresh)>0 else random.choice([0,1,2])
            logger.info('Action got (random): %s \n', action)

    return action

fwd_threshold = 0.5

def run(data_file,prev_data_file):
    global hyperparam,episode,algo_move_count,i_bumped,bump_episode,last_action,\
        fwd_threshold,num_bumps,do_train,last_persisted,visualize_filters
    global logger,logging_level,logging_format,bump_logger,prev_log_bump_ep
    global netsize_logger
    global initial_run
    global  hit_obstacle,reversed,move_complete

    logger.info('\nEPISODIC INFORMATION \n')
    logger.info('Episode: %s',episode)
    logger.info('Bump_episode: %s',bump_episode)
    logger.info('Number of Moves by the DeepRL: %s',algo_move_count)
    logger.info('Number of bumps: %s \n',num_bumps)

    # and after restarting both atrv_save_data and move_exec_robot
    if data_file[1].shape[0]>0 and initial_run:
        logger.debug('Very first run after Termination\n')
        last_action = 1
        shared_data_file = make_shared(data_file[0],data_file[1],'inputs',False)
        action = test(shared_data_file[0],1,model,hyperparam.model_type)

        algo_move_count += 1
        initial_run = False

    # any other run
    elif data_file[1].shape[0]>0:
        shared_data_file = make_shared(data_file[0],data_file[1],'inputs',False)

        logger.info("Input size: %s", shared_data_file[0].eval().shape)
        logger.info("Label size: %s", shared_data_file[1].eval().shape)
        logger.info("Count: %s", shared_data_file[2])

        if prev_data_file is not None:
            prev_shared_data_file = make_shared(prev_data_file[0],prev_data_file[1],'prev_inputs',False)
        else:
            prev_shared_data_file = None

        action = -1

        if shared_data_file and shared_data_file[2]>0 and do_train:
            if hyperparam.model_type == 'DeepRLMultiSoftmax':
                train_multi_softmax(hyperparam.batch_size, shared_data_file, prev_shared_data_file,
                           hyperparam.pre_epochs, hyperparam.finetune_epochs, hyperparam.learning_rate, model, hyperparam.model_type)
            elif hyperparam.model_type == 'LogisticRegression':
                train_logistic_regression(hyperparam.batch_size,shared_data_file,prev_shared_data_file,hyperparam.learning_rate,model,hyperparam.model_type)
            elif hyperparam.model_type == 'SDAE':
                train_sdae(hyperparam.batch_size,shared_data_file,prev_shared_data_file,hyperparam.learning_rate,model,hyperparam.model_type)
            else:
                raise NotImplementedError


            if hyperparam.model_type == 'DeepRLMultiSoftmax' or hyperparam.model_type == 'DeepRL':
                layer_out_size_str = str(episode)+',' + str(algo_move_count)+','
                for l in model.layers[:-1]:
                    layer_out_size_str += str(l.current_out_size)+','

                netsize_logger.info(layer_out_size_str)
            # if i_bumped, current data has the part that went ahead and bumped (we discard this data)
            # if i_bumped, we need to get the prediction with previous data instead of feeding current data
            if not i_bumped:
                logger.info('Did not bump, so predicting with current data\n')
                action = test(shared_data_file[0],1,model,hyperparam.model_type)
            else:
                logger.info('Bumped, so predicting with previous data\n')
                action = test(prev_shared_data_file[0],1,model,hyperparam.model_type)

            algo_move_count += 1

        elif shared_data_file and shared_data_file[2]>0 and not do_train:
            action = test(shared_data_file[0],1,model,hyperparam.model_type)
            algo_move_count += 1

    else:
        logger.warning('Incompatible action received. Sending action 1')
        action=1

    # Persisting parameters
    if persist_every>0 and algo_move_count>0 and do_train and \
                    last_persisted!=algo_move_count and algo_move_count%persist_every==0:

        logger.info('[run] Persist parameters & Filters: %s',algo_move_count)
        if hyperparam.model_type == 'DeepRL' or hyperparam.model_type == 'DeepRLMultiSoftmax':
            import copy
            updated_hparams = copy.copy(hyperparam)
            updated_hparams.hid_sizes = model.get_updated_hid_sizes()

            persist_parameters(updated_hparams,model.layers, model._controller, model.get_pool_data(), model.episode)
            if visualize_filters:
                for layer_idx in range(len(hyperparam.hid_sizes)):
                    filters = model.visualize_nodes(updated_hparams.learning_rate*2.,75,layer_idx,'sigmoid')
                    create_image_grid(filters,updated_hparams.aspect_ratio,algo_move_count,layer_idx)

        if hyperparam.model_type== 'SDAE':
            persist_parameters(hyperparam,model.layers, None, None, 0)
            if visualize_filters:
                for layer_idx in range(len(hyperparam.hid_sizes)):
                    filters = model.visualize_nodes(hyperparam.learning_rate*2.,75,layer_idx,'sigmoid')
                    create_image_grid(filters,hyperparam.aspect_ratio,algo_move_count,layer_idx)

        last_persisted = algo_move_count

    # assign last_action
    last_action = action
    logger.info('Last action: %s\n', last_action)

    # logger for number of bumps
    if algo_move_count>0 and algo_move_count % bump_count_window == 0:
        logger.debug('Printing to bump_logger: Episodes (%s-%s), Bumps %s',
                     algo_move_count,algo_move_count-bump_count_window,(num_bumps-prev_log_bump_ep))
        bump_logger.info('%s,%s',algo_move_count,(num_bumps-prev_log_bump_ep))
        prev_log_bump_ep = num_bumps

    episode += 1

    # wait till the move complete to publish action
    logger.debug('Waiting for the move to complete')
    while not move_complete:
        True
    logger.debug('Move completed. Checking for bumps')

    # check if obstacle hit
    # if obstacle hit wait for reverse
    if not hit_obstacle:
        logger.debug('Did not hit an obstacle. Executing action\n')
        action_pub.publish(action)
    else:
        logger.debug('Hit an obstacle. Waiting for reverse to complete')
        temp_i = 0
        while (not reversed) and temp_i<100:
            time.sleep(0.1)
            temp_i += 1

        logger.debug('Reverse done. Executing action\n')

        # we publish the action only if we recieved reserved signal
        # we do not publish if the while loop terminated of timeout
        if temp_i<100:
            action_pub.publish(action)




def create_image_grid(filters,aspect_ratio,fig_id,layer_idx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.cm as cm
    from math import ceil
    global dir_suffix
    # mutex is used to prevent errors raised by matplotlib (regarding main thread)

    filt_w,filt_h = int(ceil(len(filters)**0.5)),int(ceil(len(filters)**0.5))
    fig = plt.figure(1,figsize=(filt_w,filt_h)) # use the same figure, otherwise get main thread is not in main loop
    grid = ImageGrid(fig, 111, nrows_ncols=(filt_w, filt_h), axes_pad=0.1, share_all=True)

    for i in range(len(filters)):
        grid[i].set_xticks([])
        grid[i].set_yticks([])
        grid[i].set_xticklabels([])
        grid[i].set_yticklabels([])
        grid[i].imshow(filters[i].reshape(aspect_ratio[1],aspect_ratio[0]), cmap = cm.Greys_r)  # The AxesGrid object work as a list of axes.

    plt.savefig(dir_suffix+os.sep+'filters'+str(fig_id)+'-'+str(layer_idx)+'.jpg')
    plt.clf()
    plt.close('all')

prev_data = None
bump_episode = -1

import scipy.misc
def save_images_curr_prev(prev,curr):
    global episode
    global hyperparam
    prev_mat = np.reshape(prev,tuple(hyperparam.aspect_ratio[::-1]))
    curr_mat = np.reshape(curr,tuple(hyperparam.aspect_ratio[::-1]))
    scipy.misc.imsave('Images'+os.sep+'img'+str(episode)+'_1.jpg', prev_mat)
    scipy.misc.imsave('Images'+os.sep+'img'+str(episode)+'_2.jpg', curr_mat)

def callback_data_save_status(msg):

    global data_inputs,data_labels,prev_data,i_bumped,bump_episode,episode,run_mutex
    global hit_obstacle,reversed,move_complete
    global save_images
    global hyperparam

    initial_data = int(msg.data)
    logger.info('Data Received ...')
    input_count = data_inputs.shape[0]
    label_count = data_labels.shape[0]
    logger.info('currdata (before): %s, %s',data_inputs.shape,data_labels.shape)
    if data_inputs.shape[0] != data_labels.shape[0]:
        logger.warning('data and label counts are different. Matching')
        if label_count >input_count:
            for _ in range(label_count-input_count):
                data_labels = np.delete(data_labels,-1,0)
        if label_count < input_count:
            for _ in range(input_count-label_count):
                data_inputs = np.delete(data_inputs,-1,0)

    if data_inputs.shape[0]%hyperparam.batch_size!=0:
        logger.debug('Input count is not a factor of batchsize')
        num_batches = int(data_inputs.shape[0]/hyperparam.batch_size)
        logger.debug('Deleting %s inputs from total %s',data_inputs.shape[0]-num_batches*hyperparam.batch_size,data_inputs.shape[0])

        data_inputs = np.delete(data_inputs,np.s_[0:data_inputs.shape[0]-num_batches*hyperparam.batch_size],0)
        data_labels = np.delete(data_labels,np.s_[0:data_inputs.shape[0]-num_batches*hyperparam.batch_size],0)
        logger.debug('Total size after deleting %s',data_inputs.shape[0])

    # for the 1st iteration
    if i_bumped or initial_data==0:
        logger.info("Initial run after the break!")
        #prev_data = None
        bump_episode = episode-1
        i_bumped = False

    if prev_data is not None:
        logger.info('prevdata: %s, %s\n',prev_data[0].shape,prev_data[1].shape)

    logger.info('currdata (after): %s, %s',data_inputs.shape,data_labels.shape)

    if data_inputs.shape[0]>0 and data_labels.shape[0]>0:
        if save_images and (data_inputs is not None) and (prev_data is not None):
                logger.debug('Saving last image of previous and current image batches\n')
                save_images_curr_prev(prev_data[0][-1],data_inputs[-1])
        try:
            run_mutex.acquire()
            run([data_inputs,data_labels],prev_data)
            if not hit_obstacle:
                prev_data = [data_inputs,data_labels]

        finally:
            run_mutex.release()
    else:
        logger.warning("\nNo data to run\n")

    move_complete = False
    hit_obstacle = False
    reversed = False

def callback_data_inputs(msg):
    global data_inputs,hyperparam
    global logger
    data_inputs = np.asarray(msg.data,dtype=np.float32).reshape((-1,hyperparam.in_size))/255.
    data_inputs = data_inputs
    #import scipy # use if you wanna check algo receive images correctly
    #scipy.misc.imsave('rec_img'+str(episode)+'.jpg', data_inputs[-1].reshape(64,-1)*255)
    logger.info('Recieved. Input size: %s',data_inputs.shape)

def callback_data_labels(msg):
    global data_labels,out_size
    global logger

    data_labels = np.asarray(msg.data,dtype=np.int32).reshape((-1,))
    logger.info('Recieved. Label size: %s',data_labels.shape)

def callback_restored_bump(msg):
    global last_action,action_pub
    global episode,algo_move_count
    global reversed

    reversed = True
    #episode += 1
    #algo_move_count += 1
    #action_pub.publish(last_action)

def callback_obstacle_status(msg):
    global  hit_obstacle,move_complete

    hit_obstacle = True
    move_complete = True

def callback_path_finish(msg):
    global move_complete
    move_complete = True

def callback_initial_run(msg):
    global  initial_run
    initial_run = True

data_inputs = None
data_labels = None
action_pub = None

hyperparam = None
episode=0
algo_move_count = 0

do_train = 1
persist_every = 10
bump_count_window = 25
restore_last = None
last_persisted = 0

# if True, we use (not i_bumped) instances to add to pool
# if False, we use i_bumped instances to add to pool
pool_with_not_bump = True
visualize_filters = True
save_images = False

hit_obstacle = False
reversed = False
initial_run = False

logger = logging.getLogger(__name__)

dir_suffix = None
bump_logger = None
netsize_logger = None
time_logger,test_time_logger = None,None
param_logger = None

prev_log_bump_ep = 0

run_mutex = Lock()
move_complete = False

hyperparam = None

class HyperParams(object):

    def __init__(self):
        self.in_size = -1
        self.aspect_ratio = [] #w,h
        self.out_size = -1
        self.model_type = None
        self.activation = 'sigmoid' #relu/sigmoid/softplus
        self.learning_rate = -1
        self.batch_size = -1

        # number of batches we use to train the model, because the images at the end are the most important
        # so we get <train_batch_count> batches from the end of the data stream to train
        self.train_batch_count = -1

        self.epochs = -1
        self.hid_sizes = []
        self.corruption_level = -1
        self.lam = -1
        self.iterations = -1
        self.r_pool_size = -1
        self.ft_pool_size = -1
        self.pre_epochs = -1
        self.finetune_epochs = -1
        self.sim_thresh = -1
        self.multi_softmax = False
        self.dropout = 0
        self.action_frequency = 5

    def print_hyperparams(self):
        global logger

        logger.info('\nRETRIEVING HYPERPARAMETERS ...\n')
        logger.info('Model type: ' + self.model_type)
        logger.info('Activation: ' + self.activation)
        logger.info('Dropout: ' + str(self.dropout))
        logger.info('Batch size: ' + str(self.batch_size))
        logger.info('Epochs: ' + str(self.epochs))

        layers_str = str(self.in_size) + ', '
        for s in self.hid_sizes:
            layers_str += str(s) + ', '
        layers_str += str(self.out_size)
        logger.info('Network Configuration: ' + layers_str)

        logger.info('Learning Rate: ' + str(self.learning_rate))
        logger.info('Iterations: ' + str(self.iterations))
        logger.info('Lambda Regularizing Coefficient: ' + str(self.lam))
        logger.info('Pool Size (Train): ' + str(self.r_pool_size))

if __name__ == '__main__':

    global restore_last, do_train, persist_every, bump_count_window
    global logger, logging_level, logging_format, bump_logger

    import getopt
    import os
    from input_averager import InputAverager

    #logging.basicConfig(level=logging_level,format=logging_format)
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",['restore_model=',"restore_pool=","train=","persist_window=","bump_window="])
    except getopt.GetoptError as err:
        logger.error('<filename>.py --restore_model=<filename> --restore_pool=<filename> --train=<0or1> --persist_window=<int> --bump_window=<int>')
        logger.critical(err)
        sys.exit(2)

    logger.info('\nPRINTING SYSTEM ARGUMENTS ... \n')
    #when I run in command line
    restore_model, restore_pool_fn = None,None
    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--restore_model':
                logger.info('--restore_model: %s',arg)
                restore_model = arg
            if opt == '--restore_pool':
                logger.info('--restore_pool: %s',arg)
                restore_pool_fn = arg
            if opt == '--train':
                logger.info('--train: %s',bool(int(arg)))
                do_train = bool(int(arg))
            if opt == '--persist_window':
                logger.info('--persist window: %s',arg)
                persist_every = int(arg)
            if opt == '--bump_window':
                logger.info('--bump_window: %s',arg)
                bump_count_window = int(arg)

    theano.config.floatX = 'float32'

    hyperparam = HyperParams()

    if restore_model is not None and restore_pool_fn is not None:
        restore_data = pickle.load(open(restore_model, "rb"))
        restore_pool = pickle.load(open(restore_pool_fn, "rb"))
        hyperparam = restore_data[0]
        restore_model = restore_data[1:]

        logger.info('\nRESTORING FROM PERSISTED DATA ...\n')
        hyperparam.print_hyperparams()

        model = make_model(hyperparam,restore_data=restore_model,restore_pool=restore_pool)
    else:
        hyperparam.in_size = 7424
        hyperparam.aspect_ratio = [128,58]
        hyperparam.out_size = 3
        hyperparam.model_type = 'DeepRLMultiSoftmax' # DeepRLMultiSoftmax or LogisticRegression or SDAE
        hyperparam.activation = 'sigmoid'
        hyperparam.dropout = 0.
        hyperparam.learning_rate = 0.01 #0.01 multisoftmax #0.2 logistic
        hyperparam.batch_size = 5
        #hyperparam.train_batch_count = 2

        hyperparam.epochs = 1

        hyperparam.hid_sizes = [32]
        hyperparam.init_sizes = []
        hyperparam.init_sizes.extend(hyperparam.hid_sizes)

        hyperparam.corruption_level = 0.15
        hyperparam.lam = 0.1
        hyperparam.iterations = 5
        hyperparam.r_pool_size = 50
        hyperparam.ft_pool_size = 100
        hyperparam.pre_epochs = 5
        hyperparam.finetune_epochs = 1
        hyperparam.sim_thresh = 0.94
        hyperparam.multi_softmax = True
        hyperparam.action_frequency = 5

        hyperparam.print_hyperparams()
        model = make_model(hyperparam,restore_data=None,restore_pool=None)

    if do_train==1 and persist_every==-1:
        logger.warning('\nWARNING: NO SYSTEM ARGUMENTS SPECIFIED. USING DEFAULTS ...\n')

    dir_suffix = 'in'+ str(hyperparam.in_size) + '_out' + str(hyperparam.out_size)\
                  + '_type' + hyperparam.model_type + '_act-'+ hyperparam.activation \
                  + '_dropout' + str(hyperparam.dropout) +'_lr' + str(hyperparam.learning_rate)\
                  + '_batch' + str(hyperparam.batch_size) + '_hid' + '-'.join([str(h) for h in hyperparam.init_sizes])\
                  + '_sim' + str(hyperparam.sim_thresh)

    if not os.path.exists(dir_suffix):
        os.mkdir(dir_suffix)
    else:
        override_dir = raw_input('Folder already exist. Continue?(Y/N)')
        if not (override_dir == 'Y' or override_dir == 'y'):
            sys.exit(2)

    if not os.path.exists('Images'):
        os.mkdir('Images')

    param_logger = logging.getLogger('ParamLogger')
    param_logger.setLevel(logging.INFO)
    param_filename = dir_suffix + os.sep + 'param_log_train.log' if do_train else 'param_log_test' + file_suffix + '.log'
    param_fh = logging.FileHandler(param_filename)
    param_fh.setLevel(logging.INFO)
    param_fh.setFormatter(logging.Formatter())
    param_logger.addHandler(param_fh)

    param_logger.info('Input size: %s',str(hyperparam.in_size))
    param_logger.info('Input aspect ratio: %s',hyperparam.aspect_ratio)
    param_logger.info('Output size: %s',str(hyperparam.out_size))
    param_logger.info('Model type: %s',hyperparam.model_type)
    param_logger.info('Activation: %s',hyperparam.activation)
    param_logger.info('Dropout rate: %s',str(hyperparam.dropout))
    param_logger.info('Learning rate: %s',str(hyperparam.learning_rate))
    param_logger.info('Batch size: %s',str(hyperparam.batch_size))
    param_logger.info('Epochs: %s',str(hyperparam.epochs))
    param_logger.info('Hidden layers: %s',str(hyperparam.hid_sizes))
    param_logger.info('Hidden (initial) layers: %s',str(hyperparam.init_sizes))
    param_logger.info('Corruption level: %s',str(hyperparam.corruption_level))
    param_logger.info('Lambda: %s',str(hyperparam.lam))
    param_logger.info('Iterations: %s',str(hyperparam.iterations))
    param_logger.info('Recent Pool size: %s',str(hyperparam.r_pool_size))
    param_logger.info('Finetune Pool size: %s',str(hyperparam.ft_pool_size))
    param_logger.info('Pre epochs: %s',hyperparam.pre_epochs)
    param_logger.info('Finetune epochs: %s',hyperparam.finetune_epochs)
    param_logger.info('Similarity threshold: %s',hyperparam.sim_thresh)
    param_logger.info('Multi softmax: %s',hyperparam.multi_softmax)

    bump_logger = logging.getLogger('BumpLogger')
    bump_logger.setLevel(logging.INFO)
    bump_filename = dir_suffix + os.sep + 'bump_log_train.log' if do_train else 'bump_log_test' + file_suffix + '.log'
    fh = logging.FileHandler(bump_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter())
    bump_logger.addHandler(fh)

    if do_train:
        netsize_logger = logging.getLogger('NetSizeLogger')
        netsize_logger.setLevel(logging.INFO)
        netsize_filename = dir_suffix + os.sep + 'netsize_log_train.log'
        fh = logging.FileHandler(netsize_filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter())
        netsize_logger.addHandler(fh)

    time_logger = logging.getLogger('TrainTimeLogger')
    time_logger.setLevel(logging.INFO)
    time_filename = dir_suffix + os.sep + 'train_time_log_train.log'
    fh = logging.FileHandler(time_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter())
    time_logger.addHandler(fh)

    test_time_logger = logging.getLogger('TestTimeLogger')
    test_time_logger.setLevel(logging.INFO)
    test_time_filename = dir_suffix + os.sep + 'test_time_log_train.log'
    fh = logging.FileHandler(test_time_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter())
    test_time_logger.addHandler(fh)

    #batch_count = 5
    #input_avger = InputAverager(batch_count,batch_size,in_size)
    #console = logging.StreamHandler(sys.stdout)
    #console.setFormatter(logging.Formatter(logging_format))
    #console.setLevel(logging_level)
    #logger.addHandler(console)


    rospy.init_node("deep_rl_node")
    action_pub = rospy.Publisher('action_status', Int16, queue_size=10)
    rospy.Subscriber("/data_sent_status", Int16, callback_data_save_status)
    rospy.Subscriber("/data_inputs", numpy_msg(Floats), callback_data_inputs)
    rospy.Subscriber("/data_labels", numpy_msg(Floats), callback_data_labels)
    rospy.Subscriber("/restored_bump",Bool,callback_restored_bump)
    rospy.Subscriber("/obstacle_status", Bool, callback_obstacle_status)
    rospy.Subscriber("/initial_run",Bool,callback_initial_run)
    rospy.Subscriber("/autonomy/path_follower_result",Bool,callback_path_finish)

    rospy.spin()


