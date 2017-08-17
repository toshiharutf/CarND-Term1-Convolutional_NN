# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 01:34:00 2017

@author: Toshiharu
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

saver = tf.train.Saver()
save_file = './train_model.ckpt'

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))