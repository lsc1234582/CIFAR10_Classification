import tensorflow as tf
import operator
import math
from functools import reduce

def weight_variable(feature_dim, output_dim, name):
    return tf.Variable(tf.truncated_normal([feature_dim, output_dim], stddev=1.0/math.sqrt(float(feature_dim))), name=name)

def bias_variable(dim, name):
    return tf.Variable(tf.zeros([dim]), name=name)

def two_layer_softmax(input_shape=[32, 32, 3], num_output=10):
    flattened_input_shape = reduce(operator.mul, input_shape, 1)
    X = tf.placeholder(dtype='float32', shape=[None, flattened_input_shape], name='input') 
    h_layer_1_dim = 100
    with tf.name_scope('fc1'):
        W = weight_variable(flattened_input_shape, h_layer_1_dim, 'W')
        b = bias_variable(h_layer_1_dim, 'b')
        y = tf.nn.relu(tf.matmul(X, W) + b)
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        y_drop = tf.nn.dropout(y, keep_prob)
    with tf.name_scope('fc2'):
        W = weight_variable(h_layer_1_dim, num_output, 'W')
        b = bias_variable(num_output, 'b')
        y = tf.matmul(y_drop, W) + b
    return X, y, keep_prob
 
