from datetime import datetime
import math
import time
import numpy as np
# import dataset
import tensorflow.python.platform
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

vgg16_npy_path = "vgg16.npy"
data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
print("npy file loaded")

def conv(input_tensor, name, kw, kh, n_out, dw=1, dh=1, activation_fn=tf.nn.relu):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights',
                                  shape=None,
                                  dtype=tf.float32,
                                  initializer=tf.constant(data_dict[name][0]))
        biases = tf.get_variable('bias',
                                 shape=None,
                                 dtype=tf.float32,
                                 initializer=tf.constant(data_dict[name][1]))
        # weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32, xavier_initializer())
        # biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
        activation = activation_fn(tf.nn.bias_add(conv, biases))
        return activation, weights, biases

def sparse_conv(input_tensor, sparse_var, name, dw=1, dh=1, activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        indicies = tf.get_variable(name='indicies',
                                   initializer=sparse_var.indices,
                                   dtype=tf.int16)
        values = tf.get_variable(name='values',
                                 initializer=sparse_var.values,
                                 dtype=tf.float32)
        dense_shape = tf.get_variable(name='dense_shape',
                                      initializer=sparse_var.dense_shape,
                                      dtype=tf.int64)
        weights = tf.sparse_to_dense(tf.cast(indicies, tf.int64),
                                     dense_shape,
                                     values)
        bias = tf.get_variable(name='bias', initializer=sparse_var.bias)

        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
        activation = activation_fn(tf.nn.bias_add(conv, bias))

    return activation, weights, bias


def fully_connected(input_tensor, name, n_out, activation_fn=tf.nn.relu):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        if name in data_dict:
            weights = tf.get_variable('weights',
                                      shape=None,
                                      dtype=tf.float32,
                                      initializer=tf.constant(data_dict[name][0]))
            biases = tf.get_variable('bias',
                                     shape=None,
                                     dtype=tf.float32,
                                     initializer=tf.constant(data_dict[name][1]))
        else:
            weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
            biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        if activation_fn:
            return activation_fn(logits), weights, biases
        else:
            return logits, weights, biases

def sparse_fully_connected(input_tensor, sparse_var, name, activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        indicies = tf.get_variable(name='indicies',
                                   initializer=sparse_var.indices,
                                   dtype=tf.int16)
        values = tf.get_variable(name='values',
                                 initializer=sparse_var.values,
                                 dtype=tf.float32)
        dense_shape = tf.get_variable(name='dense_shape',
                                      initializer=sparse_var.dense_shape,
                                      dtype=tf.int64)
        weights = tf.sparse_to_dense(tf.cast(indicies, tf.int64),
                                     dense_shape,
                                     values)
        bias = tf.get_variable(name='bias', initializer=sparse_var.bias)

        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), bias)

        if activation_fn:
            return activation_fn(logits), weights, bias
        else:
            return logits, weights, bias


def pool(input_tensor, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name)


def loss(logits, onehot_labels):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')
    loss = tf.reduce_mean(xentropy, name='loss')
    return loss


def topK_error(predictions, labels, K=5):
    correct = tf.cast(tf.nn.in_top_k(predictions, labels, K), tf.float32)
    accuracy = tf.reduce_mean(correct)
    error = 1.0 - accuracy
    return error

def average_gradients(grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
