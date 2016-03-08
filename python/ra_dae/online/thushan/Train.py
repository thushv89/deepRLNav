

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
    if layer_params is not None:
        W,b,b_prime = layer_params[0]
        print('Restoring (i=0) W: ',W.shape)
        print('Restoring (i=0) b: ',b.shape)
        print('Restoring (i=0) b_prime: ',b_prime.shape)
        print('Restoring (i=0) init_size: ', init_sizes[0])
        layers.append(NNLayer.Layer(hparams.in_size, W.shape[1], False, W, b, b_prime,init_sizes[0]))
    else:
        layers.append(NNLayer.Layer(hparams.in_size, hparams.hid_sizes[0], False, None, None, None))

    for i, size in enumerate(hparams.hid_sizes,0):
        if i==0: continue
        if layer_params is not None:
            W,b,b_prime = layer_params[i]
            print('Restoring (i=',i,') W: ',W.shape)
            print('Restoring (i=',i,') b: ',b.shape)
            print('Restoring (i=',i,') b_prime: ',b_prime.shape)
            print('Restoring (i=',i,') init_size: ', init_sizes[i])
            layers.append(NNLayer.Layer(W.shape[0], W.shape[1], False, W, b, b_prime,init_sizes[i]))
        else:
            layers.append(NNLayer.Layer(hparams.hid_sizes[i-1], hparams.hid_sizes[i], False, None, None, None))


    layers.append(NNLayer.Layer(hparams.hid_sizes[-1], hparams.out_size, True, None, None, None))

    print('Finished Creating Layers')

    return layers

num_classes = 3
def make_model(hparams, restore_data=None,restore_pool=None):
    # restoredata should have layer
    rng = T.shared_randomstreams.RandomStreams(0)
    deeprl_ep = 0
    global episode,num_bumps,deeprl_episodes
    if restore_data is not None:
        layer_params,init_sizes,rl_q_vals,(ep,nb,deeprl_ep) = restore_data
        episode = ep
        num_bumps = nb
        policies = [RLPolicies.ContinuousState(q_vals=q) for q in rl_q_vals]
        layers = make_layers(hparams, layer_params, init_sizes)
    else:
        policies = [RLPolicies.ContinuousState() for _ in range(len(hparams.hid_sizes))]
        layers = make_layers(hparams)

    if hparams.model_type == 'DeepRL':
        model = DLModels.DeepReinforcementLearningModel(
            layers, hparams.corruption_level, rng, hparams.iterations, hparams.lam, hparams.batch_size,
            hparams.r_pool_size, hparams.ft_pool_size, policies,hparams.sim_thresh,hparams.out_size,pool_with_not_bump)
        if restore_pool is not None:
            pool,diff_pool = restore_pool
            p_data = theano.shared(np.asarray(pool[0], dtype=theano.config.floatX), 'pool' )
            p_data_y = theano.shared(np.asarray(pool[1], dtype=theano.config.floatX), 'pool_y')

            dp_data = theano.shared(np.asarray(diff_pool[0], dtype=theano.config.floatX), 'diff_pool' )
            dp_data_y = theano.shared(np.asarray(diff_pool[1], dtype=theano.config.floatX), 'diff_pool_y')

            model.restore_pool(hparams.batch_size,p_data,p_data_y,dp_data,dp_data_y)
            model.set_episode_count(deeprl_ep)

    elif hparams.model_type == 'SAE':
        model = DLModels.StackedAutoencoderWithSoftmax(
            layers,hparams.corruption_level,rng,hparams.lam,hparams.iterations)

    model.process(T.matrix('x'), T.matrix('y'))

    return model

# KEEP THIS AS 1 otherwise can lead to issues
last_action = 1 # this is important for DeepRLMultiLogreg
i_bumped = False
num_bumps = 0


def train(batch_size, data_file, prev_data_file, pre_epochs, fine_epochs, learning_rate, model, model_type):

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

        if model_type == 'DeepRL':

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
            train_adaptive,update_pool = model.train_func(
                arc, learning_rate, data_file[0], theano.shared(np.asarray(y_list,dtype=theano.config.floatX)),
                batch_size)

            check_fwd = model.check_forward(arc, data_file[0], data_file[1], batch_size)

        elif model_type == 'SAE':
            pretrain_func,finetune_func,finetune_valid_func = model.train_func(arc, learning_rate, data_file[0], data_file[1], batch_size, False, None,None)


        print('[train] Training data ...')
        try:
            if model_type == 'DeepRL':

                from collections import Counter
                global last_action,episode,i_bumped,num_bumps,pool_with_not_bump

                i_bumped = False
                alpha = 0.5

                # if True, intead of using 0.5 and 0.5 for unknown directions, use
                # (1-alpha)*current output + alpha * (0.5 output)
                use_exp_averaging = False


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

                    if pool_with_not_bump and last_action == 0:
                        print '[train] I took correct action 0 (Adding to pool)'
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([1.0,0,0])
                        _,update_pool = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            update_pool(p_t_batch)

                    elif pool_with_not_bump and last_action == 1:
                        print '[train] I took correct action 1 (Adding to pool)'
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0,1.,0])
                        _,update_pool = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            update_pool(p_t_batch)
                    elif pool_with_not_bump and last_action==2:
                        print '[train] I took correct action 2 (Adding to pool)'
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0,0,1.0])
                        _,update_pool = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y_tmp,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            update_pool(p_t_batch)
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
                    print '[train] bumped after taking action ', last_action
                    #p_for_batch = get_proba_func(t_batch)
                    #act_for_batch = np.argmax(p_for_batch,axis=0)
                    if last_action == 0:
                        # train using [0,0.5,0.5]
                        print '[train] I shouldve taken action 1 or 2'
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0,0.5,0.5])

                        y = np.asarray(y_tmp)

                        if use_exp_averaging:

                            get_proba_func = model.get_predictions_func(arc, prev_data_file[0], batch_size)

                            all_probas = None
                            for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                                probas = get_proba_func(p_t_batch)
                                if all_probas is None:
                                    all_probas = probas
                                else:
                                    all_probas = np.append(all_probas,probas,axis=0)

                            assert all_probas.shape == y.shape
                            y = (1-alpha)*all_probas + alpha*y

                        train_adaptive_prev,_ = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            train_adaptive_prev(p_t_batch)

                    elif last_action == 1:
                        # train adaptively using [0.5, 0, 0.5]
                        print '[train] I shouldve taken action 0 or 2'
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0.5,0,0.5])

                        y = np.asarray(y_tmp)

                        if use_exp_averaging:

                            get_proba_func = model.get_predictions_func(arc, prev_data_file[0], batch_size)

                            all_probas = None
                            for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                                probas = get_proba_func(p_t_batch)
                                if all_probas is None:
                                    all_probas = probas
                                else:
                                    all_probas = np.append(all_probas,probas,axis=0)

                            assert all_probas.shape == y.shape
                            y = (1-alpha)*all_probas + alpha*y

                        train_adaptive_prev,_ = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y,dtype=theano.config.floatX)), batch_size)
                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            train_adaptive_prev(p_t_batch)
                        # no point in takeing actions here, coz we've bumped

                    else:
                        print '[train] I shouldve taken action 0 or 1'
                        # train_using [0.5,0.5,0]
                        y_tmp = []
                        for i in range(prev_data_file[0].get_value().shape[0]):
                            y_tmp.append([0.5,0.5,0])

                        y = np.asarray(y_tmp)

                        if use_exp_averaging:

                            get_proba_func = model.get_predictions_func(arc, prev_data_file[0], batch_size)

                            all_probas = None
                            for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                                probas = get_proba_func(p_t_batch)
                                if all_probas is None:
                                    all_probas = probas
                                else:
                                    all_probas = np.append(all_probas,probas,axis=0)

                            assert all_probas.shape == y.shape
                            y = (1-alpha)*all_probas + alpha*y

                        train_adaptive_prev,_ = model.train_func(
                            arc, learning_rate, prev_data_file[0],
                            theano.shared(np.asarray(y,dtype=theano.config.floatX)), batch_size)

                        for p_t_batch in range(int(ceil(prev_data_file[2]*1.0/batch_size))):
                            train_adaptive_prev(p_t_batch)

        except StopIteration:
            pass

    end_time = time.clock()
    print('\n[train] Time taken for the episode: ', (end_time-start_time)/60, ' (mins)')
    return

def persist_parameters(updated_hparam,layers,policies,pools,deeprl_episodes):
    import pickle
    global episode, num_bumps # number of processed batches

    lyr_params = []
    layer_sizes = []
    for l in layers:
        W,b,b_prime = l.get_params()
        layer_sizes.append(l.W.get_value().shape)
        lyr_params.append((W.get_value(borrow=True),b.get_value(borrow=True),b_prime.get_value(borrow=True)))
    policy_Qs = []
    for p in policies:
        policy_Qs.append(p.get_Q())

    assert len(lyr_params)>0 and len(policy_Qs)>0 and len(layer_sizes)>0

    file_suffix = 'in'+ str(updated_hparam.in_size) + '_out' + str(updated_hparam.out_size)\
                  + '_type' + updated_hparam.model_type + '_lr' + str(updated_hparam.learning_rate)\
                  + '_batch' + str(updated_hparam.batch_size) + '_hid' + '-'.join([str(h) for h in updated_hparam.hid_sizes])\
                  + '_sim' + str(updated_hparam.sim_thresh)

    pickle.dump((updated_hparam,lyr_params, layer_sizes, policy_Qs,(episode,num_bumps,deeprl_episodes)),
                open( "params_"+str(num_bumps)+ file_suffix + ".pkl", "wb"))
    pickle.dump(pools,open('pools_'+str(num_bumps)+ file_suffix + '.pkl', 'wb'))
    
def test(shared_data_file_x,arc,model, model_type):
    global i_bumped,hyperparam
    if model_type == 'DeepRL':
        get_proba_func = model.get_predictions_func(arc, shared_data_file_x, hyperparam.batch_size)
        last_idx = int(ceil(shared_data_file_x.eval().shape[0]*1.0/hyperparam.batch_size))-1
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
    global hyperparam,episode,i_bumped,bump_episode,last_action,fwd_threshold,num_bumps,train_for
    print('[run] --------------- Episodic information ---------------')
    print('[run] episode: ',episode)
    print('[run] bump_episode: ',bump_episode)
    print('[run] number of bumps: ',num_bumps)
    print('[run] ----------------------------------------------------\n')
    # this part is for the very first action after bumping somewhere
    if data_file[1].shape[0]>0 and episode-1 == bump_episode:
        print('[run] very first episode after bump')
        last_action = 1
        shared_data_file = make_shared(data_file[0],data_file[1],'inputs',False)
        action = test(shared_data_file[0],1,model,hyperparam.model_type)
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

        if shared_data_file and shared_data_file[2]>0 and num_bumps<train_for:
            train(hyperparam.batch_size, shared_data_file, prev_shared_data_file,
                           hyperparam.pre_epochs, hyperparam.finetune_epochs, hyperparam.learning_rate, model, hyperparam.model_type)
            action = test(shared_data_file[0],1,model,hyperparam.model_type)

        elif shared_data_file and shared_data_file[2]>0 and num_bumps>=train_for:
            action = test(shared_data_file[0],1,model,hyperparam.model_type)

        if persist_every>0 and num_bumps>0 and num_bumps<=train_for and num_bumps%persist_every==0:
            print '[run] Persist parameters & Filters: ',num_bumps
            if hyperparam.model_type == 'DeepRL':
                import copy
                updated_hparams = copy.copy(hyperparam)
                if hyperparam.model_type:
                    updated_hparams.hid_sizes = model.get_updated_hid_sizes()
                persist_parameters(updated_hparams,model.layers, model._controller, model.get_pool_data(), model.episode)
                filters = model.visualize_nodes(updated_hparams.learning_rate,0)
                #filt_i = 0
                #for filt in filters:
                #    import scipy
                #    scipy.misc.imsave('filter'+str(filt_i)+'.jpg', filt)
                #    filt_i += 1
                create_image_grid(filters,num_bumps)

    else:
        action=1

    last_action = action
    print '[run] last action: ', last_action
    print "\n"

    episode += 1
    action_pub.publish(action)

def create_image_grid(filters,fig_id):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.cm as cm
    from math import ceil
    filt_w,filt_h = int(ceil(len(filters)**0.5)),int(ceil(len(filters)**0.5))
    fig = plt.figure(1) # use the same figure, otherwise get main thread is not in main loop
    grid = ImageGrid(fig, 111, nrows_ncols=(filt_w, filt_h), axes_pad=0.1)

    for i in range(len(filters)):
        grid[i].imshow(filters[i], cmap = cm.Greys_r)  # The AxesGrid object work as a list of axes.

    plt.savefig('filters'+str(fig_id)+'.jpg')

prev_data = None
bump_episode = -1
def callback_data_save_status(msg):
    global data_inputs,data_labels,prev_data,i_bumped,bump_episode,episode
    initial_data = int(msg.data)
    print '[callback] Running DeepRL ...'
    input_count = data_inputs.shape[0]
    label_count = data_labels.shape[0]
    print '[callback] currdata (before): ',data_inputs.shape,', ',data_labels.shape
    if data_inputs.shape[0] != data_labels.shape[0]:
        print '[callback] data and label counts are different. correcting'
        if label_count >input_count:
            for _ in range(label_count-input_count):
                data_labels = np.delete(data_labels,-1,0)
        if label_count < input_count:
            for _ in range(input_count-label_count):
                data_inputs = np.delete(data_inputs,-1,0)

    # for the 1st iteration
    if i_bumped or initial_data==0:
        print "Initial run after the break!\n"
        prev_data = None
        bump_episode = episode-1
        i_bumped = False

    if prev_data is not None:
        print '[callback] prevdata: ',prev_data[0].shape,' ,',prev_data[1].shape

    print '[callback] currdata (after): ',data_inputs.shape,' ,',data_labels.shape

    if data_inputs.shape[0]>0 and data_labels.shape[0]>0:
        run([data_inputs,data_labels],prev_data)
        prev_data = [data_inputs,data_labels]
    else:
        print "[callback] No data to run"



def callback_data_inputs(msg):
    global data_inputs
    global hyperparam
    data_inputs = np.asarray(msg.data,dtype=np.float32).reshape((-1,hyperparam.in_size))/255.
    data_inputs = data_inputs
    #import scipy # use if you wanna check algo receive images correctly
    #scipy.misc.imsave('rec_img'+str(episode)+'.jpg', data_inputs[-1].reshape(64,-1)*255)
    print 'Recieved. Input size: ',data_inputs.shape

def callback_data_labels(msg):
    global data_labels
    global out_size
    data_labels = np.asarray(msg.data,dtype=np.int32).reshape((-1,))
    print 'Recieved. Label size: ',data_labels.shape

data_inputs = None
data_labels = None
action_pub = None

hyperparam = None
episode=0
train_for = 400
persist_every = -1
restore_last = None

# if True, we use (not i_bumped) instances to add to pool
# if False, we use i_bumped instances to add to pool
pool_with_not_bump = True

class HyperParams(object):

    def __init__(self):
        self.in_size = -1
        self.out_size = -1
        self.model_type = None
        self.learning_rate = -1
        self.batch_size = -1
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


if __name__ == '__main__':

    global restore_last, train_for, persist_every

    import getopt
    import sys
    from input_averager import InputAverager

    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",['restore_model=',"restore_pool=","train_for=","persist_every="])
    except getopt.GetoptError as err:
        print '<filename>.py --restore_model=<filename> ' \
              '--restore_pool=<filename> --train_for=<int> --persist_every=<int>'
        print(err)
        sys.exit(2)

    #when I run in command line
    restore_model, restore_pool_fn = None,None
    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--restore_model':
                print('--restore_model: ',arg)
                restore_model = arg
                print('bool: ',restore_last)
            if opt == '--restore_pool':
                print('restore_pool: ',arg)
                restore_pool_fn = arg
                print('bool: ',restore_last)
            if opt == '--train_for':
                print('train_for: ',arg)
                train_for = int(arg)
            if opt == '--persist_every':
                print('persist every: ',arg)
                persist_every = int(arg)

    theano.config.floatX = 'float32'


    hyperparam = HyperParams()

    if restore_model is not None and restore_pool_fn is not None:
        restore_data = pickle.load(open(restore_model, "rb"))
        restore_pool = pickle.load(open(restore_pool_fn, "rb"))
        hyperparam = restore_data[0]
        restore_model = restore_data[1:]

        print('Restoring from stored ...')
        print('Initial sizes')
        model = make_model(hyperparam,restore_data=restore_model,restore_pool=restore_pool)
    else:
        hyperparam.in_size = 5184
        hyperparam.out_size = 3
        hyperparam.model_type = 'DeepRL'
        hyperparam.model_type = 'DeepRL'
        hyperparam.learning_rate = 0.05
        hyperparam.batch_size = 5
        hyperparam.epochs = 1
        hyperparam.hid_sizes = [100]
        hyperparam.corruption_level = 0.2
        hyperparam.lam = 0.03
        hyperparam.iterations = 3
        hyperparam.r_pool_size = 25
        hyperparam.ft_pool_size = 50
        hyperparam.pre_epochs = 5
        hyperparam.finetune_epochs = 1
        hyperparam.sim_thresh = 0.94

        model = make_model(hyperparam,restore_data=None,restore_pool=None)

    #batch_count = 5
    #input_avger = InputAverager(batch_count,batch_size,in_size)

    model_info = '\n\n----------------- Model Information -----------------\n'
    model_info += 'Model type: ' + hyperparam.model_type + '\n'
    model_info += 'Batch size: ' + str(hyperparam.batch_size) + '\n'
    model_info += 'Epochs: ' + str(hyperparam.epochs) + '\n'

    layers_str = str(hyperparam.in_size) + ', '
    for s in hyperparam.hid_sizes:
        layers_str += str(s) + ', '
    layers_str += str(hyperparam.out_size)
    model_info += 'Network Configuration: ' + layers_str + '\n'
    model_info += 'Learning Rate: ' + str(hyperparam.learning_rate) + '\n'
    model_info += 'Iterations: ' + str(hyperparam.iterations) + '\n'
    model_info += 'Lambda Regularizing Coefficient: ' + str(hyperparam.lam) + '\n'
    model_info += 'Pool Size (Train): ' + str(hyperparam.r_pool_size) + '\n'
    model_info += '------------------------------------------------------\n\n'
    print model_info

    rospy.init_node("deep_rl_node")
    action_pub = rospy.Publisher('action_status', Int16, queue_size=10)
    rospy.Subscriber("/data_sent_status", Int16, callback_data_save_status)
    rospy.Subscriber("/data_inputs", numpy_msg(Floats), callback_data_inputs)
    rospy.Subscriber("/data_labels", numpy_msg(Floats), callback_data_labels)

    rospy.spin()


