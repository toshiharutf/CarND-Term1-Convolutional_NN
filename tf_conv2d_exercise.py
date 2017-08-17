# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
"""
Setup the strides, padding and filter weight/bias such that
the output shape is (1, 2, 2, 3).

 new_height = (input_height - filter_height + 2 * P)/S + 1
 new_width = (input_width - filter_width + 2 * P)/S + 1
 
 For the 'VALID' padding, the output height and width are computed as:

out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

For the 'SAME' padding, the output height and width are computed as:

out_height = ceil(float(in_height) / float(strides[1]))
out_width  = ceil(float(in_width) / float(strides[2]))

"""
import tensorflow as tf
import numpy as np

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)


def conv2d(input):
    # Filter (weights and bias)
    # The shape of the filter weight is (height, width, input_depth, output_depth)
    # The shape of the filter bias is (output_depth,)
    # TODO: Define the filter weights `F_W` and filter bias `F_b`.
    # NOTE: Remember to wrap them in `tf.Variable`, they are trainable parameters after all.
    
    filter_height = 3
    filter_width = 3
    filter_depth = 3
    
    input_depth = 1
    
    
    F_W = tf.Variable(tf.random_normal([filter_height, filter_width, input_depth, filter_depth]))
    F_b = tf.Variable(tf.random_normal([filter_depth]))
    # TODO: Set the stride for each dimension (batch_size, height, width, depth)
    strides = [1, 1, 1, 1]
    # TODO: set the padding, either 'VALID' or 'SAME'.
    padding = 'VALID'
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
    # `tf.nn.conv2d` does not include the bias computation so we have to add it ourselves after.
    return tf.nn.conv2d(input, F_W, strides, padding) + F_b

out = conv2d(X)





