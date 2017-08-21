import tensorflow as tf
import operator
import math
from functools import reduce

def weight_variable(feature_dim, output_dim, name):
    return tf.Variable(tf.truncated_normal([feature_dim, output_dim], stddev=1.0/math.sqrt(float(feature_dim))), name=name)

def bias_variable(dim, name):
    return tf.Variable(tf.zeros([dim]), name=name)

def single_layer_softmax(input_shape=[32, 32, 3], num_output=10):
    flattened_input_shape = reduce(operator.mul, input_shape, 1)
    X = tf.placeholder(dtype='float32', shape=[None, flattened_input_shape], name='input') 
    W = weight_variable(flattened_input_shape, num_output, 'W')
    b = bias_variable(num_output, 'b')
    y = tf.matmul(X, W) + b
    return X, y
 
