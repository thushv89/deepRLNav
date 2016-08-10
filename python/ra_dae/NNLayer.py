__author__ = 'Thushan Ganegedara'

import numpy as np
import math
import theano
import theano.tensor as T

def relu(x):
    return T.switch(x > 0, x, 0*x)

class Layer(object):

    __slots__ = ['name', 'W', 'b', 'b_prime', 'idx', 'initial_size','activation','current_out_size']

    def __init__(self, input_size, output_size, zero=None, W=None, b=None, b_prime=None, init_sizes = None):
        self.name = '(%d*->%d*)' %(input_size,output_size) # %d represent input_size
        self.idx = T.ivector('idx_' + self.name)

        if zero:
            self.W = theano.shared(np.zeros((input_size, output_size), dtype=theano.config.floatX))
        elif W is not None:
            self.W = theano.shared(W)
        else:
            rng = np.random.RandomState(0)
            init = 4 * np.sqrt(6.0 / (input_size + output_size))
            initial = np.asarray(rng.uniform(low=-init, high=init, size=(input_size, output_size)), dtype=theano.config.floatX)

            # randomly initalise weights
            self.W = theano.shared(initial, 'W_' + self.name)

        self.b = theano.shared(b if b is not None else np.zeros(output_size, dtype=theano.config.floatX), 'b_' + self.name)
        self.b_prime = theano.shared(b_prime if b_prime is not None else np.zeros(input_size, dtype=theano.config.floatX), 'b\'_' + self.name)
        self.initial_size = (input_size, output_size) if init_sizes is None else init_sizes
        self.activation = None
        self.current_out_size = self.initial_size[1]

    def set_research_params(self,**params):
        if 'activation' in params:
            self.activation = params['activation']

    def output(self, x,activation=None,dropout=0,training=True):
        ''' Return the output of this layer as a MLP '''
        if activation is not None:
            if activation == 'sigmoid':
                if dropout>0. and not training:
                    return T.nnet.sigmoid(T.dot(x, self.W*(1.-dropout)) + self.b)
                return T.nnet.sigmoid(T.dot(x, self.W) + self.b)
            elif activation == 'relu':
                if dropout>0. and not training:
                    return relu(T.dot(x, self.W*(1.-dropout)) + self.b)
                else:
                    return relu(T.dot(x, self.W) + self.b)
            elif activation == 'softplus':
                if dropout>0. and not training:
                    return T.log(1+T.exp(T.dot(x,self.W*(1.-dropout))+self.b))
                else:
                    return T.log(1+T.exp(T.dot(x,self.W)+self.b))
            else:
                raise NotImplementedError

        if self.activation == 'sigmoid':
            if dropout>0. and not training:
                return T.nnet.sigmoid(T.dot(x, self.W*(1.-dropout)) + self.b)
            else:
                return T.nnet.sigmoid(T.dot(x, self.W) + self.b)
        elif self.activation == 'relu':
            if dropout>0. and not training:
                return relu(T.dot(x, self.W*(1.-dropout)) + self.b)
            else:
                return relu(T.dot(x, self.W) + self.b)
        elif self.activation == 'softplus':
            if dropout>0. and not training:
                return T.log(1+T.exp(T.dot(x, self.W*(1.-dropout)) + self.b))
            else:
                return T.log(1+T.exp(T.dot(x, self.W) + self.b))
        else:
            raise NotImplementedError

    def get_params(self):
        return self.W,self.b,self.b_prime

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
