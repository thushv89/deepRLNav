__author__ = 'thushv89'

import theano
import theano.tensor as T
import numpy as np

class InputAverager(object):

    def __init__(self,batch_count,batch_size,dims):
        self.batch_count = batch_count
        self.batch_size = batch_size
        self.dims = dims
        self.pool = theano.shared(np.empty((batch_count*batch_size, dims), dtype=theano.config.floatX), 'pool' )
        self.size = batch_count*batch_size
        self.position = 0
        self.filled_size = 0

    def add_to_pool(self,x):
        print('position before add: ',self.position)
        print('adding data of size: ', x.get_value().shape[0])
        if self.position + x.get_value().shape[0] > self.size:
            x = x[:(self.size-self.position),:]
        self.pool = T.set_subtensor(self.pool[self.position:self.position+x.shape[0],:],x)
        self.position = (self.position + x.get_value().shape[0])%self.size

        if self.filled_size<self.position:
            self.filled_size = self.position
        elif self.filled_size>self.position:
            self.filled_size = self.batch_size*self.batch_count

        print('filled_size: ',self.filled_size, ' position (after): ',self.position)

    def get_avg_input(self,X):

        self.add_to_pool(X)

        avg_X = T.fmatrix('avg_X')
        actual_batch_count = self.filled_size//self.batch_size
        print('actual # of batchs: ',actual_batch_count)

        early_batch_idc = []
        early_batch_idc.extend([i for i in range(
            np.max([0,self.position-X.get_value().shape[0]])//self.batch_size)]
                               )
        if self.filled_size != self.position:
            early_batch_idc.extend([i for i in range(self.position//self.batch_size,(self.filled_size//self.batch_size))])
        print('other batch idxs ',early_batch_idc)

        inputs_to_avg = theano.shared(np.empty((len(early_batch_idc), self.dims), dtype=theano.config.floatX), 'in_2_avg' )

        idx_tmp = 0
        for count_i in early_batch_idc:
            rnd_idx = (self.batch_size*count_i) + np.random.randint(0,self.batch_size)
            inputs_to_avg = T.set_subtensor(inputs_to_avg[idx_tmp,:],self.pool[rnd_idx,:]*(0.5/len(early_batch_idc)))
            idx_tmp += 1

        cond_avg_X = None
        if len(early_batch_idc)>0:
            others = T.sum(inputs_to_avg,axis=0)
            cond_avg_X = 0.5*avg_X+others
        else:
            cond_avg_X = avg_X

        avg_fn = theano.function(inputs=[],outputs=cond_avg_X, givens={avg_X:X})

        return np.asarray(avg_fn())