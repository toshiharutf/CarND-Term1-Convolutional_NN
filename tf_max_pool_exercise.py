# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:06:59 2017

@author: Toshiharu
"""

"""
Set the values to `strides` and `ksize` such that
the output shape after pooling is (1, 2, 2, 1).

 For the 'VALID' padding, the output height and width are computed as:

out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

For the 'SAME' padding, the output height and width are computed as:

out_height = ceil(float(in_height) / float(strides[1]))
out_width  = ceil(float(in_width) / float(strides[2]))
"""
import tensorflow as tf
import numpy as np

# `tf.nn.max_pool` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)

def maxpool(input):
    # TODO: Set the ksize (filter size) for each dimension (batch_size, height, width, depth)
    batch_size = 1
    filter_height = 3 
    filter_width = 3
    filter_depth = 1
    
    
    ksize = [batch_size,filter_height,filter_width,filter_depth]
    # TODO: Set the stride for each dimension (batch_size, height, width, depth)
    strides = [1, 1, 1, 1]
    # TODO: set the padding, either 'VALID' or 'SAME'.
    padding = 'VALID'
    # https://www.tensorflow.org/versions/r1.3/api_docs/python/tf/nn/max_pool
    return tf.nn.max_pool(input, ksize, strides, padding)
    
out = maxpool(X)