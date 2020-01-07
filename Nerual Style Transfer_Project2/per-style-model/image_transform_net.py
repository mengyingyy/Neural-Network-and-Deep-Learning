
from __future__ import division
import tensorflow as tf

def instance_norm(x, num_filters):
    epsilon = 1e-3

    shape = [num_filters]
    scale = tf.Variable(tf.ones(shape), name='scale')
    shift = tf.Variable(tf.zeros(shape), name='shift')

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    x_normed = tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    return scale * x_normed + shift

def conv2d(x, in_filter, out_filter, kernel, strides, relu=True):
    padding = kernel//2
    padded_x = tf.pad(x, [[0,0], [padding, padding], [padding, padding], [0,0]], mode='REFLECT')

    shape = [kernel, kernel, in_filter, out_filter]
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

    out = tf.nn.conv2d(padded_x, weight, strides=[1, strides, strides, 1], padding='VALID')
    out = instance_norm(out, out_filter)
    if relu:
        out = tf.nn.relu(out)
    return out

def transpose_conv2d(x, in_filter, out_filter, kernel, strides):
    shape = [kernel, kernel, out_filter, in_filter]
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

    output_shape = [tf.shape(x)[0], tf.shape(x)[1]*strides, tf.shape(x)[2]*strides, out_filter]
    out = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1])
    out = instance_norm(out, out_filter)
    return tf.nn.relu(out)

def residual(x, filters, kernel_size, strides):
    conv1 = conv2d(x, filters, filters, kernel_size, strides)
    conv2 = conv2d(conv1, filters, filters, kernel_size, strides, relu=False)

    return x + conv2

def transform(image):
    image = image/127.5 - 1
    image = tf.pad(image, [[0,0], [10,10], [10,10], [0,0]], mode='REFLECT')
    with tf.variable_scope('conv1'):
        conv1 = conv2d(image, 3, 32, 9, 1)
    with tf.variable_scope('conv2'):
        conv2 = conv2d(conv1, 32, 64, 3, 2)
    with tf.variable_scope('conv3'):
        conv3 = conv2d(conv2, 64, 128, 3, 2)

    with tf.variable_scope('residual1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('residual2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('residual3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('residual4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('residual5'):
        res5 = residual(res4, 128, 3, 1)

    with tf.variable_scope('deconv1'):
        deconv1 = transpose_conv2d(res5, 128, 64, 3, 2)
    with tf.variable_scope('deconv2'):
        deconv2 = transpose_conv2d(deconv1, 64, 32, 3, 2)
    with tf.variable_scope('convout'):
        convout = tf.tanh(conv2d(deconv2, 32, 3, 9, 1, relu=False))

    out = (convout+1)*127.5
    output = tf.slice(out, [0, 10, 10, 0], [-1, tf.shape(out)[1] - 20, tf.shape(out)[2] - 20, -1])

    return output
