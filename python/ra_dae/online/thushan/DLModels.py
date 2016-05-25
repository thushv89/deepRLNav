__author__ = 'Thushan Ganegedara'

import functools
import itertools
import theano
import theano.tensor as T
import numpy as np
import logging
import os
from math import ceil
import sys
import logging
import random

logging_level = logging.INFO
logging_format = '[%(name)s] [%(funcName)s] %(message)s'


def identity(x):
    return x

def relu(x):
    return T.switch(x > 0, x, 0*x)

def chained_output(layers, x,activation=None,dropout=0,training=True):
    '''
    This method is applying the given transformation (lambda expression) recursively
    to a sequence starting with an initial value (i.e. x)
    :param layers: sequence to perform recursion
    :param x: Initial value to start recursion
    :return: the final value (output after input passing through multiple neural layers)
    '''
    return functools.reduce(lambda acc, layer: layer.output(acc,activation,dropout,training), layers, x)

def iterations_shim(train, iterations):
    ''' Repeat calls to this function '''
    def func(i):
        for _ in range(iterations):
            result = train(i)
        return result

    return func

class Transformer(object):

    #__slots__ save memory by allocating memory only to the varibles defined in the list
    __slots__ = ['layers','arcs', '_x','_y','_logger','use_error','activation']

    def __init__(self,layers, arcs, activation, logger):
        self.layers = layers
        self._x = None
        self._y = None
        self._logger = logger
        self.arcs = arcs

        self.activation = activation

        # the last layer HAS to be SIGMOID (For SOFTMAX)
        for l in self.layers[:-1]:
                l.set_research_params(activation=self.activation)
        self.layers[-1].set_research_params(activation='sigmoid')

        if self._logger is not None:
             self._logger.debug('Setting activation type to %s '
                               'for this and all %s layers',self.activation,len(self.layers))

    def set_research_params(self,**params):
        raise NotImplementedError

    def make_func(self, x, y, batch_size, output, updates, transformed_x = identity):
        '''
        returns a Theano function that takes x and y inputs and return the given output using given updates
        :param x: input feature vectors
        :param y: labels of inputs
        :param batch_size: batch size
        :param output: the output to calculate (symbolic)
        :param update: how to get to output from input
        :return: Theano function
        '''
        idx = T.iscalar('idx')
        given = {
            self._x : transformed_x(x[idx * batch_size : (idx + 1) * batch_size]),
            self._y : y[idx * batch_size : (idx + 1) * batch_size]
        }

        return theano.function(inputs=[idx],outputs=output, updates=updates, givens=given, on_unused_input='warn')

    def process(self, x, y,training):
        '''
        Visit function with visitor pattern
        :param x:
        :param y:
        :return:
        '''
        pass

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity):
        '''
        Train the network with given params
        :param learning_rate: How fast it learns
        :param x: input feature vectors
        :param y: labels of inputs
        :param batch_size:
        :return: None
        '''
        pass

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        '''
        Validate the network with given parames
        :param x:
        :param y:
        :param batch_size:
        :return:
        '''
        pass

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        '''
        Calculate error
        :param x:
        :param y:
        :param batch_size:
        :return:
        '''
        pass


class DeepAutoencoder(Transformer):
    ''' General Deep Autoencoder '''
    def __init__(self,layers, corruption_level, rng, activation, dropout, lam=0.0):
        super(DeepAutoencoder,self).__init__(layers, 1, activation, None)
        self._rng = rng
        self._corr_level = corruption_level
        self.lam = lam

        self.theta = None
        self.cost = None
        # Need to find out what cost_vector is used for...
        self.cost_vector = None
        self.weight_regularizer = None
        self.dropout = dropout

    def process(self, x, y,training):
        self._x = x
        self._y = y
        saltpepper = True
        # encoding input
        for layer in self.layers:
            W, b_prime = layer.W, layer.b_prime

            #if rng is specified corrupt the inputs
            if self._rng:
                if not saltpepper:
                    x_tilde = self._rng.binomial(size=(x.shape[0], x.shape[1]), n=1,  p=(1 - self._corr_level), dtype=theano.config.floatX) * x
                else:
                    a = self._rng.binomial(size=(x.shape[0], x.shape[1]),p=(1 - self._corr_level),dtype=theano.config.floatX)
                    b = self._rng.binomial(size=(x.shape[0], x.shape[1]),p=0.5,dtype=theano.config.floatX)
                    c = T.eq(a, 0) * b
                    x_tilde = x * a + c

                y = layer.output(x_tilde)
            else:
                y = layer.output(x)

            if self.dropout > 0:
                y = self._rng.binomial(size=(y.shape[1],),  p=(1 - self.dropout), dtype=theano.config.floatX) * y

            x = y

        # decoding output and obtaining reconstruction
        for layer in reversed(self.layers):
            W, b_prime = layer.W, layer.b_prime
            if self.activation == 'sigmoid':
                x = T.nnet.sigmoid(T.dot(x,W.T) + b_prime)
            elif self.activation == 'relu':
                x = T.dot(x,W.T)+b_prime
            elif self.activation == 'softplus':
                x = T.log(1+T.exp(T.dot(x,W.T)+b_prime))
            else:
                raise NotImplementedError

        self.weight_regularizer = T.sum(T.sum(self.layers[0].W**2, axis=1))
        #weight_reg_coeff = (0.5**2) * self.layers[0].W.shape[0]*self.layers[0].W.shape[1]
        for i,layer in enumerate(self.layers[1:]):
            #self.weight_regularizer = T.dot(self.weight_regularizer**2, self.layers[i+1].W**2)
            self.weight_regularizer += T.sqrt(T.sum(T.sum(layer.W**2, axis=1)))
            #weight_reg_coeff += (0.1**2) * layer.W.shape[0]*layer.W.shape[1]

        # NOTE: weight regularizer should NOT go here. Because cost_vector is a (batch_size x 1) vector
        # where each element is cost for each element in the batch
        # costs
        # cost vector seems to hold the reconstruction error for each training case.
        # this is required for getting inputs with reconstruction error higher than average
        if self.activation == 'sigmoid':
            self.cost_vector = T.sum(T.nnet.binary_crossentropy(x, self._x),axis=1)
            self.cost = T.mean(self.cost_vector)
        elif self.activation == 'relu':
            self.cost_vector = T.sum((self._x - T.log(1 + T.exp(x)))**2,axis=1)
            self.cost = T.mean(self.cost_vector) #+ 1e-10 * self.weight_regularizer
        elif self.activation=='softplus':
            self.cost_vector = T.sum((self._x - x)**2,axis=1)
            self.cost = T.mean(self.cost_vector)
        else:
            raise NotImplementedError

        self.theta = [ param for layer in self.layers for param in [layer.W, layer.b, layer.b_prime]]
        # + (self.lam * self.weight_regularizer)

        return None

    def train_func(self, _, learning_rate, x, y, batch_size, transformed_x=identity):
        updates = [(param, param - learning_rate*grad) for param, grad in zip(self.theta, T.grad(self.cost,wrt=self.theta))]
        return self.make_func(x=x,y=y,batch_size=batch_size,output=self.cost, updates=updates, transformed_x=transformed_x)

    def indexed_train_func(self, arc, learning_rate, x, batch_size, transformed_x):

        nnlayer = self.layers[arc]
        # clone is used to substitute a computational subgraph
        transformed_cost = theano.clone(self.cost, replace={self._x : transformed_x(self._x)})

        # update only a set of neurons specified by index
        updates = [
            (nnlayer.W, T.inc_subtensor(nnlayer.W[:,nnlayer.idx], - learning_rate * T.grad(transformed_cost, nnlayer.W)[:,nnlayer.idx].T)),
            (nnlayer.b, T.inc_subtensor(nnlayer.b[nnlayer.idx], - learning_rate * T.grad(transformed_cost,nnlayer.b)[nnlayer.idx])),
            (nnlayer.b_prime, - learning_rate * T.grad(transformed_cost, nnlayer.b_prime))
        ]

        idx = T.iscalar('idx')
        givens = {self._x: x[idx * batch_size:(idx+1) * batch_size]}

        # using on_unused_inputs warn because, selected neurons could be "not depending on all the weights"
        # all the weights are a part of the cost. So it give an error otherwise
        return theano.function([idx,nnlayer.idx], None, updates=updates, givens=givens, on_unused_input='warn')

    def validate_func(self, _, x, y, batch_size, transformed_x=identity):
        return self.make_func(x=x,y=y,batch_size=batch_size,output=self.cost, updates=None, transformed_x=transformed_x)

    def get_hard_examples(self, _, x, y, batch_size, transformed_x=identity):
        '''
        Returns the set of training cases (above avg reconstruction error)
        :param _:
        :param x:
        :param y:
        :param batch_size:
        :return:
        '''
        # sort the values by cost and get the top half of it (above average error)
        indexes = T.argsort(self.cost_vector)[(self.cost_vector.shape[0] // 2):]
        return self.make_func(x=x, y=y, batch_size=batch_size, output=[self._x[indexes], self._y[indexes]], updates=None, transformed_x=transformed_x)

class StackedAutoencoder(Transformer):
    ''' Stacks a set of autoencoders '''
    def __init__(self, layers, corruption_level, rng, activation):
        super(StackedAutoencoder,self).__init__(layers, len(layers), activation, None)
        self._autoencoders = [DeepAutoencoder([layer], corruption_level, rng, activation) for layer in layers]

    def process(self, x, y,training):
        self._x = x
        self._y = y

        for autoencoder in self._autoencoders:
            autoencoder.process(x,y)

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity):
        return self._autoencoders[arc].train_func(0, learning_rate,x,y,batch_size, lambda x: chained_output(self.layers[:arc],transformed_x(x)))

    def validate_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._autoencoders[arc].validate_func(0,x,y,batch_size,lambda x: chained_output(self.layers[:arc],transformed_x(x)))

class Softmax(Transformer):

    def __init__(self, layers, iterations, rng, activation,dropout=0):
        super(Softmax,self).__init__(layers, 1, activation=activation,logger=None)

        self.theta = None
        self.results = None
        #self._errors = None
        self.cost_vector = None
        self.cost = None
        self.iterations = iterations
        self.last_out = None
        self.p_y_given_x = None
        self.dropout = dropout
        self.training = True
        self._rng = rng

        self.softmax_logger = logging.getLogger('Softmax'+ str(random.randint(0,1000)))
        self.softmax_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.softmax_logger.addHandler(console)

    def process(self, x, y,single_node_softmax=False,training=True):
        self._x = x
        self._y = y
        self.training = training

        self.last_out = chained_output(self.layers, x)


        self.theta = [param for layer in self.layers for param in [layer.W, layer.b]]
        #self._errors = T.mean(T.neq(self.results,y))
        if single_node_softmax:
            if self.dropout <= 0:
                self.p_y_given_x = chained_output(self.layers, self._x)
            else:
                self.softmax_logger.debug('Dropout selected')
                self.softmax_logger.debug('Is it training: %s\n',self.training)
                if self.training:
                    x_tmp = self._x
                    # until 1 before last hidden layer
                    for lyr in self.layers[:-1]:
                        x_tmp = lyr.output(x_tmp)
                        x_tmp = self._rng.binomial(size=(x_tmp.shape[1],),  p=(1 - self.dropout), dtype=theano.config.floatX) * x_tmp
                    # softmax
                    self.p_y_given_x = self.layers[-1].output(x_tmp)
                else:
                    self.p_y_given_x = chained_output(self.layers, self._x,dropout=self.dropout,training=self.training)

            self.cost_vector = T.nnet.binary_crossentropy(self.p_y_given_x,self._y)
            #-log(self.p_y_given_x)[T.eq(y,1.)]
        else:
            self.p_y_given_x = T.nnet.softmax(chained_output(self.layers, x))
            self.cost_vector = -T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]

        #self.cost_vector = T.nnet.categorical_crossentropy(self.p_y_given_x,y)
        #self.cost_vector = T.sum(T.sqrt((self.p_y_given_x-y)**2),axis=1)
        self.cost = T.mean(self.cost_vector)

        return None

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity, iterations=None):

        if iterations is None:
            iterations = self.iterations

        updates = [(param, param - learning_rate*grad) for param, grad in zip(self.theta, T.grad(self.cost,wrt=self.theta))]

        train = self.make_func(x,y,batch_size,self.cost,updates,transformed_x)
        ''' all print statements for this method returned None when I used iterations_shim'''
        ''' because func inside iteration_shim didn't return anything at the moment '''
        return iterations_shim(train, iterations)


    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self.make_func(x,y,batch_size,self.cost,None,transformed_x)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self.make_func(x,y,batch_size,self.cost,None, transformed_x)

    def get_y_labels(self, arc, x, y, batch_size, transformed_x = identity):
        return self.make_func(x, y, batch_size, self._y, None, transformed_x)

    def get_predictions_func(self, arc, x, y, batch_size, transformed_x = identity):
        idx = T.iscalar('idx')
        return theano.function(inputs=[idx], outputs=self.p_y_given_x,
                               givens={
                                   self._x : transformed_x(x[idx * batch_size : (idx + 1) * batch_size])
                               }, updates=None)
class Pool(object):

    #theano.config.compute_test_value = 'warn'
    ''' A ring buffer (Acts as a Queue) '''
    __slots__ = ['size', 'max_size', 'num_classes','position', 'data', 'data_y', '_update','pool_logger']

    def __init__(self, row_size, max_size,num_classes):
        self.size = 0
        self.max_size = max_size
        self.position = 0

        self.pool_logger = logging.getLogger('Pool'+ str(random.randint(0,1000)))
        self.pool_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.pool_logger.addHandler(console)

        self.data = theano.shared(np.empty((max_size, row_size), dtype=theano.config.floatX), 'pool' )
        self.data_y = theano.shared(np.empty((max_size,num_classes), dtype=theano.config.floatX), 'pool_y')

        x = T.matrix('new_data')
        y = T.matrix('new_data_y')
        pos = T.iscalar('update_index')

        # update statement to add new data from the position of the last data point
        update = [
            (self.data, T.set_subtensor(self.data[pos:pos+x.shape[0]],x)),
            (self.data_y, T.set_subtensor(self.data_y[pos:pos+y.shape[0]],y))]

        # function to update the data and data_y
        self._update = theano.function([pos, x, y], updates=update)

    def remove(self, idx, batch_size):
        self.pool_logger.debug('Pool size: %s',self.size)
        self.pool_logger.debug('Removing batch at %s from pool',idx)
        pool_indexes = self.as_size(self.size,batch_size)
        self.pool_logger.debug('Pool indexes %s', pool_indexes)
        self.pool_logger.debug('Max idx in pool: ', np.max(pool_indexes))
        for i in range(idx,np.max(pool_indexes)):
            self.pool_logger.debug('replacing data at %s with data at %s', i,(i+1))
            T.set_subtensor(self.data[i*batch_size:(i)*batch_size],self.data[(i+1)*batch_size:(i+1)*batch_size])
            T.set_subtensor(self.data_y[i*batch_size:(i)*batch_size],self.data_y[(i+1)*batch_size:(i+1)*batch_size])

        self.pool_logger.debug('The position was at %s', self.position)
        self.position = self.position - batch_size
        self.pool_logger.debug('Now the position at %s', self.position)
        self.size = self.position

    def add(self, x, y, rows=None):

        if not rows:
            rows = x.shape[0]

        # get the latter portion of x and y (of size max_size)
        if rows > self.max_size:
            x = x[rows - self.max_size:]
            y = y[rows - self.max_size:]

        # if new data size + current position exceed max_size
        if rows + self.position > self.max_size:
            available_size = self.max_size - self.position
            self._ring_add(x[:available_size], y[:available_size])
            x = x[available_size:]
            y = y[available_size:]

        self._ring_add(x,y)

    def add_from_shared(self, index, batch_size, x, y):
        self.add(x[index * batch_size:(index+1) * batch_size].eval(), y[index * batch_size:(index+1) * batch_size].eval(), batch_size)

    # this someway returns batch indexes
    def as_size(self, new_size, batch_size):
        batches = new_size // batch_size
        starting_index = self.position // batch_size
        index_space = self.size // batch_size
        return [(starting_index - i + index_space) % index_space for i in range(batches)]

    def clear(self):
        self.size = 0
        self.position = 0

    def _ring_add(self, x, y):
        self._update(self.position, x, y)
        self.size = min(self.size + x.shape[0], self.max_size)
        self.position = (self.position + x.shape[0]) % self.max_size

    def restore_pool(self,batch_size,x,y):
        self.add_from_shared(0,batch_size,x,y)

    def get_np_data(self):
        return self.data.get_value(),self.data_y.get_value()

class StackedAutoencoderWithSoftmax(Transformer):

    __slots__ = ['_autoencoder', '_layered_autoencoders', '_combined_objective', '_softmax', 'lam', '_updates', '_givens', 'rng', 'iterations', '_error_log','_reconstruction_log','_valid_error_log']

    def __init__(self, layers, corruption_level, rng, lam, iterations, activation):
        super(StackedAutoencoderWithSoftmax,self).__init__(layers, 1, activation=activation, logger=None)

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng, activation=activation)
        self._layered_autoencoders = [DeepAutoencoder([self.layers[i]], corruption_level, rng,activation=activation)
                                       for i, layer in enumerate(self.layers[:-1])] #[:-1] gets all items except last
        #self._softmax = Softmax(layers,iterations)
        self._softmax = CombinedObjective(layers, corruption_level, rng, lam, iterations,activation=activation)
        self.lam = lam
        self.iterations = iterations
        self.rng = np.random.RandomState(0)

        self._error_log = []
        self._reconstruction_log = []
        self._valid_error_log = []

    def process(self, x, y,training):
        self._x = x
        self._y = y

        self._autoencoder.process(x,y,training=training)
        self._softmax.process(x,y,training=training)

        for ae in self._layered_autoencoders:
            ae.process(x, y,training=training)

    def train_func(self, arc, learning_rate, x, y, v_x, v_y,batch_size, transformed_x=identity):

        layer_greedy = [ ae.train_func(arc, learning_rate, x,  y, batch_size, lambda x, j=i: chained_output(self.layers[:j], x)) for i, ae in enumerate(self._layered_autoencoders) ]
        ae_finetune_func = self._autoencoder.train_func(0, learning_rate, x, y, batch_size)
        error_func = self.error_func(arc, x, y, batch_size, transformed_x)
        reconstruction_func = self._autoencoder.validate_func(arc, x, y, batch_size, transformed_x)

        softmax_train_func = self._softmax.train_func(0,learning_rate,x,y,batch_size)
        valid_error_func = self.error_func(arc, v_x, v_y, batch_size, transformed_x)
        def pre_train(batch_id):

            for _ in range(int(self.iterations)):
                for i in range(len(self.layers)-1):
                    layer_greedy[i](int(batch_id))
                pre_cost = ae_finetune_func(batch_id)

        def finetune(batch_id):
            softmax_train_func(batch_id)
            self._valid_error_log.append(valid_error_func(batch_id))
            self._reconstruction_log.append(reconstruction_func(batch_id))
            return self._valid_error_log[-1]

        return [pre_train,finetune]

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self._softmax.validate_func(arc, x, y,batch_size)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.error_func(arc, x, y, batch_size)

    def get_y_labels(self, arc, x, y, batch_size, transformed_x = identity):
        return self.make_func(x, y, batch_size, self._y, None, transformed_x)

    def act_vs_pred_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.act_vs_pred_func(arc, x, y, batch_size, transformed_x)

class MergeIncrementingAutoencoder(Transformer):

    __slots__ = ['_autoencoder', '_layered_autoencoders', '_combined_objective', '_softmax', 'lam', '_updates', '_givens', 'rng', 'iterations','minc_logger']

    def __init__(self, layers, corruption_level, rng, lam, iterations, activation,dropout):
        self.minc_logger = logging.getLogger('MINC-AE'+ str(random.randint(0,1000)))
        self.minc_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.minc_logger.addHandler(console)

        super(MergeIncrementingAutoencoder,self).__init__(layers, 1, activation=activation, logger=self.minc_logger)

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng, activation=activation,dropout=dropout)
        self._layered_autoencoders = [DeepAutoencoder([self.layers[i]], corruption_level, rng, activation=activation,dropout=dropout)
                                       for i, layer in enumerate(self.layers[:-1])] #[:-1] gets all items except last
        self._softmax = Softmax(layers,iterations,rng,activation=activation,dropout=dropout)
        self._combined_objective = CombinedObjective(layers, corruption_level, rng, lam, iterations,activation=activation,dropout=dropout)
        self.lam = lam
        self.iterations = iterations
        self.rng = np.random.RandomState(0)

    def process(self, x, y,single_node_softmax,training):
        self._x = x
        self._y = y
        self._autoencoder.process(x,y,training=training)
        self._softmax.process(x,y,single_node_softmax,training=training)
        self._combined_objective.process(x,y,single_node_softmax,training=training)
        for ae in self._layered_autoencoders:
            ae.process(x,y,training=training)

    def merge_inc_func(self, learning_rate, batch_size, x, y):

        m = T.fmatrix('m')
        # map operation applies a certain function to a sequence. This is the upper part of cosine dist eqn
        m_dists, _ = theano.map(lambda v: T.sqrt(T.dot(v, v.T)), m)
        # dimshuffle(0,'x') is converting N -> Nx1
        m_cosine = (T.dot(m, m.T)/m_dists) / m_dists.dimshuffle(0,'x')

        # T.tri gives a matrix with 1 below diagonal (including diag) and zero elsewhere
        # flatten() gives a one dim tensor having all the data in original
        # finfo gives the maximum value of floats
        # seems np.finfo(theano.config.floatX).max is used to make result
        # it is unwise to use something like m_cosine * T.tri instead of this approach,
        # as if some cosine_dist are zero they will be confused with the other zero items (resulted zeros of multiplication) in the argsort
        m_ranks = T.argsort((m_cosine - T.tri(m.shape[0]) * np.finfo(theano.config.floatX).max).flatten())[(m.shape[0] * (m.shape[0]+1)) // 2:]

        # function for getting the merge scores (cosine distance) for neurons
        score_merges = theano.function([m], m_ranks)

        # greedy layer-wise training
        layer_greedy = [ae.indexed_train_func(0, learning_rate, x, batch_size, lambda  x, j=i: chained_output(self.layers[:j], x)) for i, ae in enumerate(self._layered_autoencoders)]
        # fintune is done by optimizing cross-entropy between x and reconstructed_x
        finetune = self._autoencoder.train_func(0, learning_rate, x, y, batch_size)


        # set up cost function
        mi_cost = self._softmax.cost + self.lam * self._autoencoder.cost

        all_mi_train_funcs = []
        for j in range(len(self._autoencoder.layers)):
            mi_updates = []

            # calculating merge_inc updates
            # increment a subtensor by a certain value
            for i, nnlayer in enumerate(self._autoencoder.layers):
                # do the inc_subtensor update only for the first layer
                # update the rest of the layers normally
                if i == j:
                    # removed ".T" in the T.grad operation. It seems having .T actually
                    # causes a dimension mismatch
                    mi_updates += [ (nnlayer.W, T.inc_subtensor(nnlayer.W[:,nnlayer.idx],
                                    - learning_rate * T.grad(mi_cost, nnlayer.W)[:,nnlayer.idx])) ]
                    mi_updates += [ (nnlayer.b, T.inc_subtensor(nnlayer.b[nnlayer.idx],
                                    - learning_rate * T.grad(mi_cost,nnlayer.b)[nnlayer.idx])) ]
                else:
                    mi_updates += [(nnlayer.W, nnlayer.W - learning_rate * T.grad(mi_cost, nnlayer.W))]
                    mi_updates += [(nnlayer.b, nnlayer.b - learning_rate * T.grad(mi_cost,nnlayer.b))]

                mi_updates += [(nnlayer.b_prime, -learning_rate * T.grad(mi_cost,nnlayer.b_prime))]

            softmax_theta = [self.layers[-1].W, self.layers[-1].b]

            mi_updates += [(param, param - learning_rate * grad)
                           for param, grad in zip(softmax_theta, T.grad(mi_cost, softmax_theta))]

            idx = T.iscalar('idx')


            given = {
                self._x : x[idx*batch_size : (idx+1) * batch_size],
                self._y : y[idx*batch_size : (idx+1) * batch_size]
            }
            # this is to test after implementing DeepRLMultiSoftmax
            '''print 'i:',i,',j:',j
            test_given = {
                self._x : x[idx*batch_size : (idx+1) * batch_size],
                self._y : y[idx*batch_size : (idx+1) * batch_size]
            }
            test_given_y = {
                self._y : y[idx*batch_size : (idx+1) * batch_size]
            }
            test_y_eq_1 = theano.function([idx],T.eq(self._y,1),givens=test_given_y)
            print(test_y_eq_1(0))
            test_mi_train = theano.function([idx],self._softmax.cost,givens=test_given)
            print(test_mi_train(0))'''

            mi_train = theano.function([idx, self.layers[j].idx], mi_cost, updates=mi_updates, givens=given)
            all_mi_train_funcs.append(mi_train)

        # the merge is done only for the first hidden layer.
        # apperantly this has been tested against doing this for all layers.
        # but doing this only to the first layer has achieved the best performance
        def merge_model(pool_indexes, merge_percentage, inc_percentage,layer_idx):
            '''
            Merge/Increment the batch using given pool of data
            :param pool_indexes:
            :param merge_percentage:
            :param inc_percentage:
            :return:
            '''

            reg_added_node_idx = []
            reg_removed_node_idx = []

            prev_map = {}
            #bottom_dimensions = self.layers[layer_idx].initial_size[0]

            used = set()
            empty_slots = []

            # first layer
            layer_weights = self.layers[layer_idx].W.get_value().T.copy()
            layer_bias = self.layers[layer_idx].b.get_value().copy()
            bottom_dimensions = layer_weights.shape[1]
            # initialization of weights
            init = 4 * np.sqrt(6.0 / (sum(layer_weights.shape)))

            # number of nodes to merge or increment
            merge_count = int(merge_percentage * self.layers[layer_idx].initial_size[1])
            inc_count = int(inc_percentage * self.layers[layer_idx].initial_size[1])


            self.minc_logger.debug("RETREIVING HYPER PARAMETERS OF Layer %s \n", layer_idx)
            self.minc_logger.debug("Layer weights (Transpose): %s",layer_weights.shape)
            self.minc_logger.debug("Layer bias: %s",layer_bias.shape)
            self.minc_logger.debug("Layer\'s bottom size: %s",bottom_dimensions)
            self.minc_logger.debug("Merge count: %s",merge_count)
            self.minc_logger.debug("Increment count: %s",inc_count)


            # if there's nothing to merge or increment
            if merge_count == 0 and inc_count == 0:
                return

            # get the nodes ranked highest to lowest (merging order)
            #print('Nodes to merge: ', score_merges(layer_weights))

            for index in score_merges(layer_weights):
                # all merges have been performed or mergecount is zero
                if len(empty_slots) == merge_count:
                    break

                # x and y coordinates created out of index (assume these are the two nodes
                # to merge)
                x_i, y_i = index % layer_weights.shape[0], index // layer_weights.shape[0]
                #print('index: ',index,' x_i: ',x_i,' y_i: ',y_i)

                # if x_i and y_i are not in "used"`  list
                # 'used' contains merged nodes
                # 'empty_slots' are the remaining nodes after the merge
                if x_i not in used and y_i not in used:
                    # update weights and bias with avg
                    layer_weights[x_i] = (layer_weights[x_i] + layer_weights[y_i])/2
                    layer_bias[x_i] = (layer_bias[x_i] + layer_bias[y_i])/2

                    #add it to the used list
                    used.update([x_i,y_i])
                    empty_slots.append(y_i)

                    self.minc_logger.debug("x_i,y_i: %s,%s",x_i,y_i)
                    self.minc_logger.debug("Changing weight & bias at %s with (x_i+y_i)/2",x_i)

            self.minc_logger.debug('\n')
            #print('used: ',used)
            #print('empty_slots: ',empty_slots)

            #get the new size of layer
            new_size = layer_weights.shape[0] + inc_count - len(empty_slots)
            current_size = layer_weights.shape[0]

            self.minc_logger.debug("current -> new (size): %s -> %s",current_size,new_size)

            # if new size is less than current... that is reduce operation
            if new_size < current_size:
                # non_empty_slots represent the nodes that were merged
                non_empty_slots = sorted(list(set(range(0,current_size)) - set(empty_slots)), reverse=True)[:len(empty_slots)]
                #print('non empty: ',non_empty_slots)
                # prev_map, merge_node -> deleted_node
                prev_map = dict(zip(empty_slots, non_empty_slots))
                #print('prev_map: ',prev_map)
                for dest, src in prev_map.items():
                    layer_weights[dest] = layer_weights[src]
                    layer_weights[src] = np.asarray(self.rng.uniform(low=init, high=init, size=layer_weights.shape[1]), dtype=theano.config.floatX)

                    self.minc_logger.debug("Replacing weight at %s (%s) with %s (%s)",
                                  dest,layer_weights[dest].shape,src,layer_weights[src].shape)
                    if layer_idx == len(self.layers)-2:
                        reg_removed_node_idx.append(src)

                self.minc_logger.debug("Layer weights (transpose) after replacement: %s \n",layer_weights.shape)
                empty_slots = []
            # increment operation
            else:
                prev_map = {}

            # new_size: new layer size after increment/reduce op
            # prev_dimension: size of input layer
            new_layer_weights = np.zeros((new_size,bottom_dimensions), dtype = theano.config.floatX)
            self.minc_logger.info('Old layer %s size: %s',layer_idx,layer_weights.shape)
            self.minc_logger.info('New layer %s size: %s',layer_idx,new_layer_weights.shape)

            # the existing values from layer_weights copied to new layer weights
            # and it doesn't matter if layer_weights.shape[0]<new_layer_weights.shape[0] it'll assign values until it reaches the end
            new_layer_weights[:layer_weights.shape[0], :layer_weights.shape[1]] = layer_weights[:new_layer_weights.shape[0], :new_layer_weights.shape[1]]

            self.minc_logger.debug("Layer weights (after Action): %s",new_layer_weights.shape)
            self.minc_logger.debug("Copied all the weights from old matrix to new matrix\n")

            # get all empty_slots that are < new_size  +  list(prev_size->new_size)
            empty_slots = [slot for slot in empty_slots if slot < new_size] + list(range(layer_weights.shape[0],new_size))
            new_layer_weights[empty_slots] = np.asarray(self.rng.uniform(low=-init, high=init, size=(len(empty_slots), bottom_dimensions)), dtype=theano.config.floatX)

            # fills missing entries with zero
            self.minc_logger.info('Old layer %s size (bias): %s',layer_idx,layer_bias.shape)
            layer_bias.resize(new_size, refcheck=False)
            self.minc_logger.info('New layer %s size (bias): %s',layer_idx,layer_bias.shape)

            layer_bias_prime = self.layers[0].b_prime.get_value().copy()
            layer_bias_prime.resize(bottom_dimensions)

            prev_dimensions = new_layer_weights.shape[0]

            self.layers[layer_idx].W.set_value(new_layer_weights.T)
            self.layers[layer_idx].b.set_value(layer_bias)
            self.layers[layer_idx].b_prime.set_value(layer_bias_prime)

            #if empty_slots:
            #    for _ in range(int(self.iterations)):
            #        for i in pool_indexes:
            #            layer_greedy[0](i, empty_slots)

            top_layer_weights = self.layers[layer_idx+1].W.get_value().copy()

            self.minc_logger.debug("Layer weights top size: %s",top_layer_weights.shape)
            self.minc_logger.debug("Layer weights top row size: %s",prev_dimensions)
            for dest, src in prev_map.items():
                top_layer_weights[dest] = top_layer_weights[src]
                top_layer_weights[src] = np.zeros(top_layer_weights.shape[1])

            self.minc_logger.debug("Layer weights top before resize: %s",top_layer_weights.shape)
            top_layer_weights.resize((prev_dimensions, self.layers[layer_idx+1].W.get_value().shape[1]),refcheck=False)

            self.minc_logger.debug("Layer weights top after resize: %s \n",top_layer_weights.shape)

            if layer_idx != len(self.layers)-1:
                top_layer_prime = self.layers[layer_idx+1].b_prime.get_value().copy()
                top_layer_prime.resize(prev_dimensions, refcheck=False)
                self.layers[layer_idx+1].b_prime.set_value(top_layer_prime)

            self.layers[layer_idx+1].W.set_value(top_layer_weights)

            # finetune with supervised
            if empty_slots:
                if layer_idx == len(self.layers)-2:
                    reg_added_node_idx.extend(empty_slots)

                for _ in range(self.iterations):
                    for i in pool_indexes:
                        all_mi_train_funcs[layer_idx](i, empty_slots)

            return reg_added_node_idx,reg_removed_node_idx

        return merge_model


class CombinedObjective(Transformer):

    def __init__(self, layers, corruption_level, rng, lam, iterations, activation, dropout):
        super(CombinedObjective,self).__init__(layers, 1, activation=activation,logger=None)

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng,activation=activation,dropout=dropout)
        self._softmax = Softmax(layers,1, rng,activation=activation,dropout=dropout)
        self.lam = lam
        self.iterations = iterations
        self.cost = None

    def process(self, x, yy,single_node_softmax,training):
        self._x = x
        self._y = yy

        self._autoencoder.process(x,yy,training=training)
        self._softmax.process(x,yy,single_node_softmax,training=training)

        self.cost = self._softmax.cost + self.lam * self._autoencoder.cost

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity, iterations=None):

        if iterations is None:
            iterations = self.iterations

        #combined_cost = self._softmax.cost + self.lam * 0.5

        theta = []
        for layer in self.layers[:-1]:
            theta += [layer.W, layer.b, layer.b_prime]
        theta += [self.layers[-1].W, self.layers[-1].b] #softmax layer

        update = [(param, param - learning_rate * grad) for param, grad in zip(theta, T.grad(self.cost,wrt=theta))]
        comb_obj_finetune_func = self.make_func(x, y, batch_size, self.cost, update, transformed_x)
        return iterations_shim(comb_obj_finetune_func, iterations)

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self._softmax.validate_func(arc, x, y, batch_size, transformed_x)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.error_func(arc, x, y, batch_size, transformed_x)

    def get_y_labels(self, act, x, y, batch_size, transformed_x = identity):
        return self.make_func(x, y, batch_size, self._y, None, transformed_x)

    def act_vs_pred_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.act_vs_pred_func(arc, x, y, batch_size, transformed_x)

    def get_predictions_func(self, arc, x,y, batch_size, transformed_x = identity):
        return self._softmax.get_predictions_func(arc, x, y, batch_size, transformed_x)

from Train import HyperParams
class DeepReinforcementLearningModel(Transformer):

    def __init__(self, layers, rng, controller, hparam, num_classes):

        self.deeprl_logger = logging.getLogger('DeepRL'+str(random.randint(0,1000)))
        self.deeprl_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.deeprl_logger.addHandler(console)

        super(DeepReinforcementLearningModel,self).__init__(layers, 1, activation=hparam.activation,logger=self.deeprl_logger)

        self._mi_batch_size = hparam.batch_size
        self._controller = controller
        self._autoencoder = DeepAutoencoder(layers[:-1], hparam.corruption_level, rng, activation=hparam.activation,dropout=hparam.dropout)
        self._softmax = CombinedObjective(layers, hparam.corruption_level, rng, lam=hparam.lam, iterations=hparam.iterations, activation=hparam.activation, dropout=hparam.dropout)
        self._merge_increment = MergeIncrementingAutoencoder(layers, hparam.corruption_level, rng, lam=hparam.lam, iterations=hparam.iterations, activation=hparam.activation,dropout=hparam.dropout)

        # _pool : has all the data points
        # _hard_pool: has data points only that are above average reconstruction error
        self._pool = Pool(layers[0].initial_size[0], hparam.r_pool_size,num_classes)
        self._hard_pool = Pool(layers[0].initial_size[0], hparam.r_pool_size,num_classes)
        self._diff_pool = Pool(layers[0].initial_size[0], hparam.ft_pool_size,num_classes)

        self._rng = rng
        self.corruption_levels = hparam.corruption_level
        self.iterations = hparam.iterations
        self.lam = hparam.lam
        self.simi_thresh = hparam.sim_thresh
        self.train_distribution = []
        self.pool_distribution = []
        self.num_classes = num_classes
        self._error_log = []
        self._valid_error_log = []
        self._reconstruction_log = []
        self._neuron_balance_log = []
        self._network_size_log = []

        self.neuron_balance = [1 for _ in range(len(self.layers[:-1]))]

        self.episode = 0
        self.mean_batch_pool = []

        self.pool_with_not_bump = True
        self.single_node_softmax = False
        self.test_mode = False

        self.action_frequency = hparam.action_frequency
        self.action_iteration = 0

    def set_research_params(self,**params):
        self.deeprl_logger.debug('RETRIEVING RESEARCH PARAMETERS\n')
        self.deeprl_logger.debug(params)

        if 'pool_with_not_bump' in params:
            self.pool_with_not_bump = params['pool_with_not_bump']
        if 'single_node_softmax' in params:
            self.single_node_softmax = params['single_node_softmax']
        if 'test_mode' in params:
            self.test_mode = params['test_mode']

    def set_episode_count(self,val):
        self.episode = val

    def restore_pool(self,batch_size,x,y,dx,dy):
        self._pool.restore_pool(batch_size,x,y)
        self._diff_pool.restore_pool(batch_size,dx,dy)

    def get_updated_hid_sizes(self):
        new_hid_sizes = []
        for l in self.layers[:-1]:
            new_hid_sizes.append(l.W.get_value().shape[1])
        return new_hid_sizes

    def process(self, x, y,training):
        self._x = x
        self._y = y
        self._autoencoder.process(x, y,training=training)
        self._softmax.process(x, y,self.single_node_softmax,training=training)
        self._merge_increment.process(x, y,self.single_node_softmax,training=training)


    def pool_if_different(self, pool, batch_id, batch_size,x, y):

        self.deeprl_logger.debug('Pool size (before): %s',pool.size)
        def magnitude(x):
            '''  returns sqrt(sum(v(i)^2)) '''
            return sum([v **2 for v in x]) ** 0.5

        #this method works as follows. Assum a 3 label case
        #say x = '0':5, '1':5
        #say y = '0':2, '1':3, '2':5
        #then the calculation is as follows
        #for every label that is in either x or y (i.e. '0','1','2')
        #xval,yval = that val if it exist else 0
        #top accumulate xval*yval

        def get_mean(b_id):
            idx = T.iscalar('idx')
            sym_x = T.matrix('x')
            mean_func = theano.function(inputs=[idx],outputs=T.mean(sym_x,axis=0),updates=None,
                                        givens={sym_x:x[idx*batch_size:(idx+1)*batch_size]})
            return mean_func(b_id).tolist()

        def compare(x,y):
            '''  Calculate Cosine similarity between x and y '''
            top = 0

            for k in range(len(x)):
                xval, yval = x[k], y[k]
                top += xval * yval

            return top / (magnitude(x) * magnitude(y))


        # the below statement get the batch scores, batch scores are basically
        # the cosine distance between a given batch and the current batch (last)
        # for i in range(-1,-1 - batches_covered) gets the indexes as minus indices as it is easier way to count from back of array


        if len(self.mean_batch_pool)>0:

            #print('pool dist: ')
            #for i,dist in enumerate(pool_dist):
            #    print(i,': ',dist,'\n')

            batch_scores = [(i, compare(get_mean(batch_id), self.mean_batch_pool[i])) for i in range(len(self.mean_batch_pool))]
            # mean is the mean cosine score
            #mean = np.mean([ v[1] for v in batch_scores ])
            #print('Batch Scores ...')
            #print(batch_scores)
            self.deeprl_logger.debug('Max Similarity (Current vs Pool): %.5f', np.max([s[1] for s in batch_scores]))
            # all non_station experiments used similarity threshold 0.7
            if np.max([s[1] for s in batch_scores]) < self.simi_thresh:
                self.deeprl_logger.debug('Adding batch %s to pool', batch_id)
                if len(self.mean_batch_pool) == pool.max_size/batch_size:
                    self.mean_batch_pool.pop(0)
                self.mean_batch_pool.append(get_mean(batch_id))
                pool.add_from_shared(batch_id, batch_size, x, y)

            #random batch switch. this was introduced hoping it would help stationary situation. if does not,
            # comment it. all non_stationary once did not use this part
            # stationary use 0.99 threshold, non-station 0.9
            '''if np.max([s[1] for s in batch_scores]) > 0.9 and np.random.random()<0.1:
                max_idx = np.argmax([s[1] for s in batch_scores])
                print('random: switching batch at ', max_idx)
                dist_val = pool_dist.pop(max_idx)

                pool.remove(max_idx,batch_size)
                pool_dist.append(dist_val)
                pool.add_from_shared(batch_id,batch_size,x,y)'''

        else:
            self.deeprl_logger.debug('Pool is empty. adding batch %s to pool', batch_id)
            self.mean_batch_pool.append(get_mean(batch_id))
            pool.add_from_shared(batch_id, batch_size, x, y)




    # pool_relevent pools all the batches from the current to the last batch that satisfies
        # cosine_dist(batch) < mean
    def pool_relevant(self, pool, distribution, batch_size, current=None):

        if current == None:
            current = distribution[-1]

        def magnitude(x):
            '''  returns sqrt(sum(v(i)^2)) '''
            return sum((v **2 for v in x.values() )) ** 0.5

        def compare(x,y):
            '''  Calculate Cosine distance between x and y '''
            top = 0

            for k in set(x) | set(y):
                xval, yval = x[k] if k in x else 0, y[k] if k in y else 0
                top += xval * yval

            return top / (magnitude(x) * magnitude(y))

        # score over batches for this pool
        # used to get only the distributions of the batches actually in the pool
        batches_covered = int(pool.size // batch_size)

        # the below statement get the batch scores, batch scores are basically
        # the cosine distance between a given batch and the current batch (last)
        # for i in range(-1,-1 - batches_covered) gets the indexes as minus indices as it is easier way to count from back of array


        batch_scores = [(i % batches_covered, compare(current, distribution[i])) for i in range(-1,-1 - batches_covered)]
        # mean is the mean cosine score
        mean = np.mean([ v[1] for v in batch_scores ])

        # takewhile(predicate, iterable) returns elements until the predicate is true
        # get all the batches with batch score greater than mean
        last = [0, 0]
        for last in itertools.takewhile(lambda s: s[1] > mean, batch_scores):
            pass

        return 1 - (last[0] / batches_covered)

    def get_batch_count(self, data_y):
        from collections import Counter
        dist = Counter(data_y)
        norm_dist = {str(k): v / sum(dist.values()) for k, v in dist.items()}
        return norm_dist

    def train_func(self, arc, learning_rate, x, y, batch_size, apply_x=identity):

        train_func = self._softmax.train_func(arc, learning_rate, x, y, batch_size, apply_x)

        reconstruction_func = self._autoencoder.validate_func(arc, x, y, batch_size, apply_x)
        error_func = self.error_func(arc, x, y, batch_size, apply_x)
        valid_error_func = self.error_func(arc,x,y, batch_size, apply_x)

        merge_inc_func_batch = self._merge_increment.merge_inc_func(learning_rate, self._mi_batch_size, x, y)
        merge_inc_func_pool = self._merge_increment.merge_inc_func(learning_rate, self._mi_batch_size, self._pool.data, self._pool.data_y)
        merge_inc_func_hard_pool = self._merge_increment.merge_inc_func(learning_rate, self._mi_batch_size, self._hard_pool.data, self._hard_pool.data_y)

        hard_examples_func = self._autoencoder.get_hard_examples(arc, x, y, batch_size, apply_x)

        train_func_pool = self._softmax.train_func(arc, learning_rate, self._pool.data, self._pool.data_y, batch_size, apply_x)
        train_func_hard_pool = self._softmax.train_func(arc, learning_rate, self._hard_pool.data, self._hard_pool.data_y, batch_size, apply_x)
        train_func_diff_pool = self._softmax.train_func(arc, learning_rate, self._diff_pool.data, self._diff_pool.data_y, batch_size, apply_x)

        if self.test_mode:
            idx = T.iscalar('test_idx')
            test_y_1_indices = theano.function([idx],T.eq(self._y,1.),givens={self._y:y[idx*batch_size:(idx+1)*batch_size]})
            test_pYgivenX = theano.function([idx],self._softmax._softmax.p_y_given_x,givens={self._x:x[idx*batch_size:(idx+1)*batch_size]})
            test_output = theano.function([idx],chained_output(self.layers, self._x),givens={self._x:x[idx*batch_size:(idx+1)*batch_size]})

        def train_pool(pool, pool_func, amount):
            self.deeprl_logger.info('\nTRAINING WITH POOL (SHUFFLED) OF SIZE %s \n',int(pool.size))
            pool_indexes = pool.as_size(int(pool.size * amount), batch_size)
            np.random.shuffle(pool_indexes)
            for i in pool_indexes:
                pool_func(i)

        def moving_average(log, n):

            weights = np.exp(np.linspace(-1, 0, n))
            weights /= sum(weights)
            return np.convolve(log, weights)[n-1:-n+1]

        def moving_average_v2(log,n):
            weights = np.exp(np.linspace(-1, 0, n))
            weights /= sum(weights)
            weights_rev = weights[::-1]
            conv = np.convolve(log,weights_rev)
            return conv[int(len(conv)/2)]

        # get early stopping
        def train_adaptively(batch_id):
            from math import sqrt


            # For Test purpose only
            if self.test_mode:
                self.deeprl_logger.debug('Testing y==1 indices in batch %s',batch_id)
                self.deeprl_logger.debug(test_y_1_indices(batch_id).flatten().tolist())

                self.deeprl_logger.debug('Testing P(Y|X) in batch %s',batch_id)
                self.deeprl_logger.debug(test_pYgivenX(batch_id).flatten().tolist())

                self.deeprl_logger.debug('Testing chained_output for batch %s',batch_id)
                self.deeprl_logger.debug(test_output(batch_id).flatten().tolist())

            self._error_log.append(np.asscalar(error_func(batch_id)))
            self.deeprl_logger.debug('Softmax Error for batch %s: %.4f',batch_id,self._error_log[-1])
            self._valid_error_log.append(np.asscalar(valid_error_func(batch_id)))
            err_for_layers = [self._valid_error_log[-1]]

            rec_err = reconstruction_func(batch_id)
            self._reconstruction_log.append(np.asscalar(rec_err))
            self.deeprl_logger.debug('Reconstruction Error for batch %s: %.4f', batch_id, rec_err)

            self._neuron_balance_log.append(self.neuron_balance)

            self._pool.add_from_shared(batch_id, batch_size, x, y)
            #self._hard_pool.add(*hard_examples_func(batch_id))

            #print('size before pool_if_diff: ',self._diff_pool.size)
            if not self.pool_with_not_bump:
                self.pool_if_different(self._diff_pool,batch_id, batch_size, x, y)
            elif self.pool_with_not_bump and self.episode == 0:
                self.pool_if_different(self._diff_pool,batch_id, batch_size, x, y)

            #print('size after pool_if_diff: ',self._diff_pool.size)

            #print('[train_adaptively] self.pool: ',self._pool.size, ',', self._pool.position, ',', self._pool.max_size)
            #print('[train_adaptively] self.diff_pool: ',self._diff_pool.size, ',', self._diff_pool.position, ',', self._diff_pool.max_size)
            #print('[train_adaptively] self.pool (after): ',self._pool.data.get_value().shape[0], ',', self._pool.data_y.eval().shape[0])

            data = {
                'mea_15': moving_average_v2(self._error_log, 15),
                'mea_10': moving_average_v2(self._error_log, 10),
                'mea_5': moving_average_v2(self._error_log, 5),
                'pool_relevant': self.pool_relevant(self._pool,self.train_distribution,batch_size),
                'initial_size': [l.initial_size[1] for l in self.layers[:-1]],
                'input_size':self.layers[0].initial_size[0],
                'hard_pool_full': self._hard_pool.size == self._hard_pool.max_size,
                'error_log': self._error_log,
                'valid_error_log': self._valid_error_log,
                'curr_error': err_for_layers[-1],
                'neuron_balance': self._neuron_balance_log[-1],
                'reconstruction': self._reconstruction_log[-1],
                'r_15': moving_average_v2(self._reconstruction_log, 15)
            }

            def merge_increment(func, pool, amount, merge, inc,layer_idx):

                #nonlocal neuron_balance
                change = 1 + inc - merge #+ 0.05 * ((self.layers[1].W.get_value().shape[0]/self.layers[1].initial_size[0])-2.)

                self.deeprl_logger.debug('Neuron balance (prev=>current) for layer %s: %.3f => %.3f',
                              layer_idx, self.neuron_balance[layer_idx], self.neuron_balance[layer_idx] * change)
                self.neuron_balance[layer_idx] *= change

                add_idx,rem_idx = func(pool.as_size(int(pool.size * amount), self._mi_batch_size), merge, inc, layer_idx)
                return add_idx,rem_idx

            #this is where reinforcement learning comes to play
            if self.episode%self.action_frequency==0:

                funcs = {
                    'merge_increment_pool' : functools.partial(merge_increment, merge_inc_func_pool, self._pool),
                    'merge_increment_hard_pool': functools.partial(merge_increment, merge_inc_func_hard_pool, self._hard_pool),
                    'pool': functools.partial(train_pool, self._pool, train_func_pool),
                    'pool_finetune':functools.partial(train_pool, self._diff_pool, train_func_diff_pool),
                    'hard_pool': functools.partial(train_pool, self._hard_pool, train_func_hard_pool),
                    'hard_pool_clear': self._hard_pool.clear,
                }

                for ctrl_i in range(len(self._controller)):

                    self._controller[ctrl_i].move(self.action_iteration, data, funcs, ctrl_i)
                    err_for_layers.append(np.asscalar(valid_error_func(batch_id)))

                    #update 'curr_error'
                    data['curr_error'] = err_for_layers[-1]

                    action,change = self._controller[ctrl_i].get_current_action_change()
                    print '%s,%s'%(action,change)
                    if action=="Action.increment":
                        self.layers[ctrl_i].current_out_size += change
                        print "new size current out size %s" %self.layers[ctrl_i].current_out_size
                    elif action == 'Action.reduce':
                        self.layers[ctrl_i].current_out_size -= change
                        print "new size current out size %s" %self.layers[ctrl_i].current_out_size
                    elif action=='None' or action == 'Action.pool':
                        self.layers[ctrl_i].current_out_size += 0
                    else:
                        raise NotImplementedError

                self.action_iteration += 1


            self.deeprl_logger.info("\nTRAINING FOR EPISODE: %s\n",self.episode)
            self.deeprl_logger.debug("Errors for layers: %s", err_for_layers)
            self.deeprl_logger.debug("Neuron balance: %s", self.neuron_balance)
            str_size = str(self.layers[0].W.get_value().shape[0])
            for l in self.layers:
                str_size += " => " + str(l.W.get_value().shape[1])
            self.deeprl_logger.info(str_size+'\n')


            train_func(batch_id)

            #self._network_size_log.append(self.layers[0].W.get_value().shape[1])
            self.episode += 1


        def update_pool(batch_id):
            self.pool_if_different(self._diff_pool,batch_id, batch_size, x, y)

        return train_adaptively,update_pool

    def visualize_nodes(self,learning_rate,iterations,layer_idx,activation,use_scan=False):
        in_size = self.layers[0].W.get_value().shape[0]

        def in_bounds(x):
            if np.sqrt(np.sum(x**2))>1:
                return False
            else:
                return True
        if not use_scan:
            idx = T.iscalar('idx')
            max_inputs = []
            for n in range(self.layers[layer_idx].W.get_value().shape[1]):
                if activation == 'sigmoid':
                    n_max_input = theano.shared(np.random.rand(in_size)*(1.0/in_size),'max_input')
                elif activation == 'relu':
                    n_max_input = theano.shared(0.5 + np.random.rand(in_size)*0.4,'max_input')
                elif activation == 'softplus':
                    n_max_input = theano.shared(np.random.rand(in_size)*(1.0/in_size),'max_input')
                else:
                    raise NotImplementedError

                h_out = chained_output(self.layers[:layer_idx+1],n_max_input,activation=activation)[idx]
                theta = [n_max_input]
                updates = [(param, param + learning_rate * grad) for param,grad in zip(theta,T.grad(h_out,wrt=theta))]
                max_in_function = theano.function(inputs=[idx],outputs=h_out,updates=updates)

                curr_hout = 0
                for it in range(iterations):
                    if in_bounds(n_max_input.get_value()):
                        curr_hout = max_in_function(n)
                        if curr_hout>0.999:
                            break
                    else:
                        break

                self.deeprl_logger.debug('stopping after %s iteration for %s hidden unit (%2.4f)',it,n,curr_hout)
                max_in = n_max_input.get_value()
                max_in = max_in/np.max(max_in)
                max_inputs.append(max_in)
        else:
            raise NotImplementedError
        return max_inputs


    def get_pool_data(self):
        return self._pool.get_np_data(),self._diff_pool.get_np_data()

    def update_train_distribution(self, t_distribution):
        self.train_distribution.append(t_distribution)

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self._softmax.validate_func(arc, x, y,batch_size)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.error_func(arc, x, y, batch_size)

    def get_y_labels(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.get_y_labels(arc, x, y, batch_size)

    def act_vs_pred_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.act_vs_pred_func(arc, x, y, batch_size, transformed_x)

    def get_predictions_func(self, arc, x, batch_size, transformed_x = identity):
        return self._softmax.get_predictions_func(arc, x, None, batch_size, transformed_x)

    def get_param_values_func(self):
        params = []
        for i,layer in enumerate(self.layers):
            params.append([layer.W.get_value(),layer.b.get_value(),layer.b_prime.get_value()])
        return params

    def check_forward(self,arc, x, y, batch_size, transformed_x = identity):
        idx = T.iscalar('idx')
        sym_y = T.ivector('y_deeprl')

        forward_func = theano.function([idx],sym_y,givens={
            sym_y:y[idx * batch_size : (idx + 1) * batch_size]
        })

        def check_forward_func(batch_id):
            tmp_out = forward_func(batch_id)
            if tmp_out[-1]==0:
                return False
            else:
                return True

        return check_forward_func

class DeepReinforcementLearningModelMultiSoftmax(object):

    def __init__(self, layers, rng, controller,hparam, num_classes):

        self.deepms_logger = logging.getLogger('DeepRLMultiSoftmax')
        self.deepms_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.deepms_logger.addHandler(console)

        #super(DeepReinforcementLearningModelMultiSoftmax,self).__init__(layers, 1, self.deepms_logger)

        #self.layers = layers
        self._mi_batch_size = hparam.batch_size
        # _pool : has all the data points
        # _hard_pool: has data points only that are above average reconstruction error

        self.layers = layers
        self._rng = rng
        self.corruption_levels = hparam.corruption_level
        self.iterations = hparam.iterations
        self.lam = hparam.lam
        self.simi_thresh = hparam.sim_thresh
        self.train_distribution = []
        self.pool_distribution = []
        self.num_classes = num_classes
        self._error_log = []
        self._valid_error_log = []
        self._reconstruction_log = []
        self._neuron_balance_log = []
        self._network_size_log = []
        self.dropout = hparam.dropout
        self.neuron_balance = [1 for _ in range(len(layers[:-1]))]

        self._controller = controller

        self.episode = 0
        self.mean_batch_pool = []
        self.pool_with_not_bump = True
        self.single_node_softmax = True

        self._softmax = []
        self._merge_increment = []

        self.deepRL_set = []

        self.deepms_logger.info('BUILDING MULTI SOFTMAX LAYERS ...\n')
        self.deepms_logger.debug('Creating 1 node softmax layers per each action')

        for n in range(self.num_classes):
            self.deepms_logger.debug('\tSoftmax layer for action %s',n)
            multi_softmax_layers = []
            for l in layers[:-1]:
                multi_softmax_layers.append(l)
            multi_softmax_layers.append(layers[-1][n])
            self.deepRL_set.append(
                DeepReinforcementLearningModel(
                    multi_softmax_layers,self._rng,controller[n],hparam,1
                )
            )
            self.deepRL_set[-1].set_research_params(pool_with_not_bump=False,single_node_softmax=True,test_mode=False)

        self.deepms_logger.debug('CHECKING IF LAYERS SETUP CORRECTLY (DeepRLMultiSoftmax) ...')

        for ae_i in range(len(self.deepRL_set[0].layers[:-1])):
            id_list_ae = [id(deeprl.layers[ae_i]) for deeprl in self.deepRL_set]
            id_0 = id(self.deepRL_set[0].layers[ae_i])
            assert all(x==id_0 for x in id_list_ae)

            self.deepms_logger.debug('\tSame Autoencoder layer is used for all DeepRLs layer %s',ae_i)

        id_list_softmax = [id(deeprl.layers[-1]) for deeprl in self.deepRL_set[1:]]
        id_softmax_0 = id(self.deepRL_set[0].layers[-1])
        assert any(x != id_softmax_0 for x in id_list_softmax)

        self.deepms_logger.debug('Different Softmax layers are used for each DeepRLs\n')


    def set_episode_count(self,val):
        self.episode = val

    def set_research_params(self,**params):
        raise NotImplementedError

    def restore_pool(self,batch_size,X,Y,DX,DY):
        i = 0
        for x,y,dx,dy in zip(X,Y,DX,DY):
            self.deepRL_set[i].restore_pool(batch_size,x,y,dx,dy)

    def get_updated_hid_sizes(self):
        return self.deepRL_set[0].get_updated_hid_sizes()

    def process(self, x, y,training):
        for drl in self.deepRL_set:
            drl.process(x,y,training=training)

    def train_func(self, arc, learning_rate, x, y, batch_size, drl_id, apply_x=identity):
        return self.deepRL_set[drl_id].train_func(arc, learning_rate, x, y, batch_size)

    def rectify_multi_softmax_layers(self,drl_id):
        self.deepms_logger.debug('Retrieving nodes added/removed from DeepRL %s',drl_id)
        add_idx,rem_idx = self.deepRL_set[drl_id]._controller[-1].add_idx, self.deepRL_set[drl_id]._controller[-1].rem_idx

        if add_idx>0 or rem_idx>0:
            drl_indices = [a for a in range(self.num_classes)]
            del drl_indices[drl_id]

            for i in drl_indices:
                self.deepms_logger.debug('Rectifying softmax layer of deepRL %s, (Add/Rem) Sizes %s/%s',i,len(add_idx),len(rem_idx))
                if len(add_idx)>0:
                    self.deepms_logger.debug('Adding new weights')
                    self.change_weights_at_idx(i,add_idx,self.deepRL_set[drl_id].layers[-1].W,True)
                elif len(rem_idx)>0:
                    self.deepms_logger.debug('Removing new weights')
                    self.change_weights_at_idx(i,rem_idx,self.deepRL_set[drl_id].layers[-1].W,False)

        #reset the add_idx,rem_idx of RLPolicy else if action_frequency>1 might cause problems
        self.deepRL_set[drl_id]._controller[-1].add_idx, self.deepRL_set[drl_id]._controller[-1].rem_idx=0,0

    def get_pool_data(self):
        pools = []
        for drl in self.deepRL_set:
            pools.append(drl.get_pool_data())

        return pools

    def update_train_distribution(self, t_distribution,drl_id):
        self.deepRL_set[drl_id].update_train_distribution(t_distribution)

    def get_predictions_func(self, arc, x, batch_size, transformed_x = identity):

        funcs = []
        for drl in self.deepRL_set:
            tmp = drl.get_predictions_func(arc, x, batch_size, transformed_x)
            funcs.append(tmp)
        return funcs

    def get_param_values_func(self):
        params = []
        for i,layer in enumerate(self.layers[:-1]):
            params.append([layer.W.get_value(),layer.b.get_value(),layer.b_prime.get_value()])

        multi_softmax = []
        for layer in self.layers[-1]:
            multi_softmax.append([layer.W.get_value(),layer.b.get_value(),layer.b_prime.get_value()])
        params.append(multi_softmax)
        return params

    def change_weights_at_idx(self, drl_id,nn_idx,drl_W,inc=False):

        new_W = None
        nnlayer = self.deepRL_set[drl_id].layers[-1]
        self.deepms_logger.debug("Current size: %s", nnlayer.W.shape)
        if inc:
            self.deepms_logger.debug('Got Increment operation')
            self.deepms_logger.debug('Adding %s weights',len(nn_idx))
            new_W = np.append(nnlayer.W.get_value(),drl_W.get_value()[nn_idx,:],axis=0)
        else:
            sorted_idx = sorted(nn_idx,reverse=True)
            new_W = nnlayer.W.get_value().copy()
            for temp_i in sorted_idx:
                new_W = np.delete(new_W,temp_i,axis=0)

            self.deepms_logger.debug('Got Reduce operation')
            self.deepms_logger.debug('Remove %s weights',len(nn_idx))


        nnlayer.W.set_value(new_W)
        self.deepms_logger.debug('Shape of new W: %s',nnlayer.W.get_value().shape)

    def check_forward(self,arc, x, y, batch_size, transformed_x = identity):
        idx = T.iscalar('idx')
        sym_y = T.ivector('y_deeprl')

        forward_func = theano.function([idx],sym_y,givens={
            sym_y:y[idx * batch_size : (idx + 1) * batch_size]
        })

        def check_forward_func(batch_id):
            tmp_out = forward_func(batch_id)
            if tmp_out[-1]==0:
                return False
            else:
                return True

        return check_forward_func

    def visualize_nodes(self,learning_rate,iterations,layer_idx,use_scan=False):
        return self.deepRL_set[0].visualize_nodes(learning_rate,iterations,layer_idx,use_scan)


class MergeIncDAE(Transformer):

    def __init__(self, layers, corruption_level, rng, iterations, lam, mi_batch_size, pool_size,num_classes,activation):

        super(MergeIncDAE,self).__init__(layers, 1, True)
        self._mi_batch_size = mi_batch_size

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng, activation=activation)
        self._softmax = CombinedObjective(layers, corruption_level, rng, lam=lam, iterations=iterations, activation=activation)
        self._merge_increment = MergeIncrementingAutoencoder(layers, corruption_level, rng, lam=lam, iterations=iterations, activation=activation)

        # _pool : has all the data points
        # _hard_pool: has data points only that are above average reconstruction error
        self._pool = Pool(layers[0].initial_size[0], pool_size*3,num_classes)
        self._hard_pool = Pool(layers[0].initial_size[0], pool_size,num_classes)
        self._pre_train_pool = Pool(layers[0].initial_size[0], 12000,num_classes)
        self._pre_train_done = False

        self.iterations = iterations
        self.lam = lam
        self.num_classes = num_classes

        self._error_log = []
        self._valid_error_log = []
        self._reconstruction_log = []

        self._neuron_balance_log = []
        self._network_size_log = []

        self._inc_log = [100]
        self.total_err = 0.
        #self.total_merge = 0.

    def process(self, x, y):
        self._x = x
        self._y = y
        self._autoencoder.process(x, y)
        self._softmax.process(x, y)
        self._merge_increment.process(x, y)

    def train_func(self, arc, learning_rate, x, y, v_x, v_y, batch_size, apply_x=identity):
        batch_pool = Pool(self.layers[0].initial_size[0], batch_size,self.num_classes)

        layer_greedy = [ ae.train_func(arc, learning_rate, self._pre_train_pool.data,  self._pre_train_pool.data_y, batch_size, lambda x, j=i: chained_output(self.layers[:j], x)) for i, ae in enumerate(self._merge_increment._layered_autoencoders) ]
        ae_finetune_func = self._autoencoder.train_func(0, learning_rate, x, y, batch_size)

        train_func = self._softmax.train_func(arc, learning_rate, x, y, batch_size, apply_x)
        train_func_pre_train = self._softmax.train_func(arc, learning_rate, self._pre_train_pool.data,  self._pre_train_pool.data_y, batch_size, apply_x)

        reconstruction_func = self._autoencoder.validate_func(arc, x, y, batch_size, apply_x)
        error_func = self.error_func(arc, x, y, batch_size, apply_x)
        valid_error_func = self.error_func(arc, v_x, v_y, batch_size, apply_x)

        merge_inc_func_hard_pool = self._merge_increment.merge_inc_func(learning_rate, self._mi_batch_size, self._hard_pool.data, self._hard_pool.data_y)

        hard_examples_func = self._autoencoder.get_hard_examples(arc, self._pool.data, self._pool.data_y, batch_size, apply_x)

        def train_pool(pool, pool_func, amount):

            for i in pool.as_size(int(pool.size * amount), batch_size):
                pool_func(i)

        def train_mergeinc(batch_id):

            rec_err = reconstruction_func(batch_id)
            self._reconstruction_log.append(np.asscalar(rec_err))
            self._error_log.append(error_func(batch_id))
            self._valid_error_log.append(valid_error_func(batch_id))

            batch_pool.add_from_shared(batch_id, batch_size, x, y)
            self._pool.add_from_shared(batch_id, batch_size, x, y)

            if self._pre_train_pool.size < self._pre_train_pool.max_size and not self._pre_train_done:
                print('Adding batch to pre-training pool')
                self._pre_train_pool.add_from_shared(batch_id, batch_size, x, y)
                return self._valid_error_log[-1]

            elif self._pre_train_pool.size == self._pre_train_pool.max_size and not self._pre_train_done:
                print('Pre training ...')
                pre_train_pool_indexes = self._pre_train_pool.as_size(int(self._pre_train_pool.size * 1), batch_size)
                for pool_idx in pre_train_pool_indexes:
                    print('\tPre training pool ', pool_idx)
                    for _ in range(int(self.iterations)):
                        for i in range(len(self.layers)-1):
                            layer_greedy[i](int(pool_idx))
                print('Fine tuning ...')
                for pool_idx in pre_train_pool_indexes:
                    print('\tFine tuning using pool ', pool_idx)
                    train_func_pre_train(pool_idx)
                print('Pre training finished')
                self._pre_train_done = True
                return self._valid_error_log[-1]

            x_hard, y_hard = hard_examples_func(batch_id)
            self._hard_pool.add(x_hard,y_hard)

            print('X indexes Size: ', x_hard.shape[0], ' Hard pool size: ', self._hard_pool.size)
            if self._hard_pool.size >= self._hard_pool.max_size:
                pool_indexes = self._hard_pool.as_size(int(self._hard_pool.size * 1), self._mi_batch_size)
                if len(self._valid_error_log)>=40:
                    eps1 = 0.025
                    eps2 = 0.01
                    curr_err = np.sum([self._valid_error_log[i] for i in range(-20,0)])
                    prev_err = np.sum([self._valid_error_log[i] for i in range(-40,-20)])
                    print('Curr Err: ', curr_err, ' Prev Err: ',prev_err)
                    if (curr_err/prev_err) < 1. - eps1:
                        print('e ratio < 1-eps',curr_err/prev_err,' ',1-eps1)
                        inc = self._inc_log[-1] + 30.
                    elif (curr_err/prev_err) > 1. - eps2:
                        print('e ratio > 1-eps',curr_err/prev_err,' ',1-eps2)
                        inc = self._inc_log[-1]/2
                    else:
                        inc = self._inc_log[-1]

                    self._inc_log.append(int(inc))
                    print('Inc Log: ',self._inc_log)
                else:
                    inc = self._inc_log[-1]

                inc_prec = inc/self.layers[1].W.get_value().shape[0]
                print('Total inc: ', inc_prec, ' Total merge: ', 0.2*inc_prec)

                merge_inc_func_hard_pool(pool_indexes, 0.2*inc_prec, inc_prec)
                self._hard_pool.clear()

            # generative error optimization
            #for _ in range(int(self.iterations/2)):
            #    for i in range(len(self.layers)-1):
            #        layer_greedy[i](int(batch_id))
            #    ae_finetune_func(batch_id)

            # discriminative error optimization
            train_func(batch_id)

            #self._network_size_log.append(self.layers[0].W.get_value().shape[1])

            return self._valid_error_log[-1]

        return train_mergeinc
    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self._softmax.validate_func(arc, x, y,batch_size)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.error_func(arc, x, y, batch_size)
