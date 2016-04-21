__author__ = 'Thushan Ganegedara'

from enum import IntEnum
from collections import defaultdict
from sklearn.gaussian_process import GaussianProcess
import numpy as np
import json
import random
import logging
import sys

logging_level = logging.INFO
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

class Controller(object):

    def move(self, i, data, funcs):
        pass

    def end(self):
        return []


class Action(IntEnum):
    pool = 1
    reduce = 2
    increment = 3

    def __repr__(self):
        return str(self)

class ContinuousState(Controller):

    __slots__ = ['learning_rate', 'discount_rate', 'prev_state', 'prev_action', 'q', 'start_time', 'action_log','rl_logger']


    def __init__(self, learning_rate=0.5, discount_rate=0.9, time_limit=1,q_vals=None):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.prev_state = None
        self.prev_action = None
        self.prev_time = 0

        if q_vals is None:
            self.q = defaultdict(dict)
        else:   # if q_vals is not null, load q from passed data
            self.q = q_vals

        self.time_limit = time_limit
        self.undesired_state = False

        self.add_idx,self.rem_idx = [],[]

        self.rl_logger = logging.getLogger(__name__+str(random.randint(0,1000)))
        self.rl_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.rl_logger.addHandler(console)
        self.gps = {}

    def move(self, i, data, funcs,layer_idx):

        # add_idx,rem_idx are the number of nodes to add/remove to the last layer of deepRL
        # the used to rectify the multiple softmax layers when one softmax add/remove neurons
        # else there will be a dimension mismatch issue
        #resetting add_idx and rem_idx every episode
        self.add_idx,self.rem_idx = [],[]

        verbose = True
        pooling = True
        q_calc_thresh = 15
        even_chose_thresh = 45
        balance_thresh = 25
        gp_fit_frequency = 10

        err_t = data['curr_error']
        err_t_minus_1 = np.mean(data['valid_error_log'][-6:-1])

        self.rl_logger.info("\nRUNNING RL for LAYER: %s, EPISODE: %s",layer_idx,i)

        #if we haven't completed 30 iterations, keep pooling
        if i <=q_calc_thresh:
            if pooling:
                funcs['pool'](1)
            return

        #what does this method do?
        def ma_state(name):
            retVal = 0
            if not len(data[name]) < 2:
                retVal = data[name][-1] - data[name][-2]

            return retVal
            #return 0 if len(data[name]) < 2 else data[name][-1] - data[name][-2]

        state = (data['r_15'][-1], data['neuron_balance'][layer_idx], ma_state('mea_15'))
        if verbose:
            self.rl_logger.debug('Current state %.3f, %.3f, %.3f' % (data['r_5'][-1], data['neuron_balance'][layer_idx], ma_state('mea_15')))

        err_diff = err_t - err_t_minus_1
        curr_err = err_t
        if verbose:
            self.rl_logger.info('Err log (t-1)->(t): %.3f -> %.3f',err_t_minus_1,err_t)
        # since we have a continuous state space, we need a regression technique to get the
        # Q-value for prev state and action for a discrete state space, this can be done by
        # using a hashtable Q(s,a) -> value


        # self.q is like this there are 3 main items Action.pool, Action.reduce, Action.increment
        # each action has (s, q) value pairs
        # use this (s, q) pairs to predict the q value for a new state
        if i%gp_fit_frequency==0:
            for a, value_dict in self.q.items():
                #makes sure, if to use regression element Q(s,a) should have atleast 3 samples
                if len(value_dict) < 2:
                    continue

                x, y = zip(*value_dict.items())

                gp = GaussianProcess(theta0=0.1, thetaL=0.001, thetaU=1, nugget=0.1)
                gp.fit(np.array(x), np.array(y))
                self.gps[a] = gp

        if self.prev_state or self.prev_action:

            #reward = - data['error_log'][-1]

            #reward = (1 - err_diff)*(-curr_err)
            if self.undesired_state:
                reward = -10
            else:
                reward = -curr_err

            neuron_penalty = 0

            if data['neuron_balance'][layer_idx] > balance_thresh or data['neuron_balance'][layer_idx] < 1:
                # the coeff was 2.0 before
                # all tests were done with 1.5
                neuron_penalty = .5 * abs(1.5 - data['neuron_balance'][layer_idx])
            reward -= neuron_penalty

            if verbose:
                self.rl_logger.debug('Reward: %10.3f, Penalty: %10.3f', reward, neuron_penalty)            #sample = reward
            #sample = reward + self.discount_rate * max(self.q[state, a] for a in self.actions)
            # len(gps) == 0 in the first time move() is called
            if len(self.gps) == 0:
                sample = reward
            else:
                sample = reward + self.discount_rate * max((np.asscalar(gp.predict([self.prev_state])[0])) for gp in self.gps.values())

            if self.prev_state in self.q[self.prev_action]:
                self.q[self.prev_action][self.prev_state] = (1 - self.learning_rate) * self.q[self.prev_action][self.prev_state] + self.learning_rate * sample
            else:
                self.q[self.prev_action][self.prev_state] = sample

        #action = list(self.Action)[i % len(self.Action)]
        #the first time move() is called
        #Evenly chosing actions is important because with this algorithm will learn q values for all actions
        if len(self.gps) == 0 or i <= even_chose_thresh:
            action = list(Action)[i % len(Action)]
            if verbose:
                self.rl_logger.info("Evenly chose action: %s", action)
        else:
            # determine best action by sampling the GPs
            if random.random() <= 0.1:
                action = list(Action)[i % len(Action)]
                self.rl_logger.info("Explore action: %s", action)
            else:
                action = max((np.asscalar(gp.predict(state)[0]), action) for action, gp in self.gps.items())[1]
                self.rl_logger.info("Chose with Q: %s", action)

            for a, gp in self.gps.items():
                self.rl_logger.debug("Approximated Q values for (above State,%s) pair: %10.3f",a, np.asscalar(gp.predict(state)[0]))

        self.undesired_state = False
        #to_move = (data['initial_size'] * 0.1) / (data['initial_size'] * data['neuron_balance'])
        #to_move = 0.25 * np.exp(-(data['neuron_balance']-1.)**2/2.) * (1. + err_diff) * (1. + curr_err)
        # newer to_move eqn

        #to_move = (50.0/data['initial_size'][layer_idx])*np.exp(-(data['neuron_balance'][layer_idx]-(.5*balance_thresh))**2/balance_thresh) * np.abs(err_diff)

        # skewed normal distribution
        #mu,sigma,alpha_skew = 1.0,5.0,4.0
        #nb = (data['neuron_balance'][layer_idx] - mu)/sigma**2
        #nb_alpha = alpha_skew*nb/2**0.5
        #to_move = (50.0/data['initial_size'][layer_idx]) * (1/(2*np.pi)**0.5) * np.exp(-nb**2/2.0) *\
        #          (1 + np.sign(nb_alpha) * np.sqrt(1-np.exp(-nb_alpha**2*((4/np.pi)+0.14*nb_alpha**2)/(1+0.14*nb_alpha**2)))) * \
        #          (1+np.abs(err_diff))
        #to_move = (100.0/data['initial_size'][layer_idx]) * np.abs(err_diff)


        to_move = 5.0/data['initial_size'][layer_idx]
        self.rl_logger.info('Nodes to add/remove: %10.3f', to_move*data['initial_size'][layer_idx])

        if to_move<0.1/data['initial_size'][layer_idx]:
            self.rl_logger.debug('Nodes to add/remove negligable')
            if pooling:
                funcs['pool'](1)
                action = Action.pool

        else:
            if data['neuron_balance'][layer_idx] < 10.0/data['initial_size'][layer_idx] and action == Action.reduce:
                self.rl_logger.debug('Undesired State. Neuron balance too low (%10.3f)... Executing pool operation',data['neuron_balance'][layer_idx])
                funcs['pool'](1)
                self.undesired_state = True


            elif pooling and action == Action.pool:
                funcs['pool_finetune'](1)
                self.add_idx,self.rem_idx = [],[]
            elif action == Action.reduce:
                # method signature: amount, to_merge, to_inc
                # introducing division by two to reduce the impact
                self.add_idx,self.rem_idx = funcs['merge_increment_pool'](data['pool_relevant'], to_move/2., 0,layer_idx)
            elif action == Action.increment:
                self.add_idx,self.rem_idx = funcs['merge_increment_pool'](data['pool_relevant'], 0, to_move,layer_idx)

        self.prev_action = action
        self.prev_state = state


    def get_Q(self):
        return self.q

    def end(self):
        return [{'name': 'q_state.json', 'json': json.dumps({str(k):{str(tup): value for tup, value in v.items()} for k,v in self.q.items()})}]