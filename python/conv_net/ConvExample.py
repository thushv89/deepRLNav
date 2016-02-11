__author__ = 'Thushan Ganegedara'

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy as np

rng = np.random.RandomState(23462)

input = T.tensor4(name='input')

# 2 = num of feature maps in destination
# 3 = num of feature maps in source (input) RGB colors
# 9x9 = receptive field connected to dest from source
w_shp = (2,3,9,9)
w_bound = np.sqrt(3*9*9)
W = theano.shared(np.asarray(
    rng.uniform( low=-1.0/w_bound, high= 1./w_bound,
                 size=w_shp),
    dtype=input.dtype),name='W')

# 2 biases for each dest feature map
b_shp = (2,)
b = theano.shared(np.asarray(
    rng.uniform(low=-.5, high=.5, size=b_shp),
    dtype=input.dtype),name='b')

conv_out = conv.conv2d(input,W)

# dimshuffle(x,0,x,x) make b (n) to b (1xnx1x1)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))

f = theano.function([input],output)

import pylab
from PIL import Image

img = Image.open('3wolfmoon.jpg')
img = np.asarray(img,dtype='float64')/256.
img_ =  img.transpose(2,0,1).reshape(1,3,639,516)
filtered_img = f(img_)

pylab.subplot(1,3,1); pylab.axis('off');pylab.imshow(img)
pylab.gray();

pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()