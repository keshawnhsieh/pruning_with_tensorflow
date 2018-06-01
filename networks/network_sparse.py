from typing import Union

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from networks.network_dense import FullyConnectedClassifier
from utils import tensorflow_utils
from utils import pruning_utils

import layers as L

class FullyConnectedClassifierSparse(FullyConnectedClassifier):

    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 sparse_layers: list,
                 model_path: str,
                 activation_fn=tf.nn.relu,
                 scope='FullyConnectedClassifierSparse',
                 verbose=True):

        self.input_size = input_size
        self.n_classes = n_classes
        self.sparse_layers = sparse_layers
        self.model_path = model_path
        self.activation_fn = activation_fn
        self.scope = scope
        self.verbose = verbose

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(self.scope):

                self._create_placeholders()

                self.logits = self._build_network(inputs=self.inputs,
                                                  sparse_layers=self.sparse_layers,
                                                  activation_fn=self.activation_fn)

                self.loss = self._create_loss(logits=self.logits,
                                              labels=self.labels)

                self._create_metrics(logits=self.logits,
                                     labels=self.labels,
                                     loss=self.loss)

                self.saver = self._create_saver(tf.global_variables())
                self.init_variables(tf.global_variables())

                if self.verbose:
                    print('\nSuccessfully created graph for {model}.'.format(
                                                            model=self.scope))
                    print('Number of parameters (four bytes == 1 parameter): {}.\n'.format(
                        pruning_utils.calculate_number_of_sparse_parameters(
                                                            self.sparse_layers)))

    def _create_placeholders(self):
    
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=(None, 224, 224, 3),
                                     name='inputs')
    
        self.labels = tf.placeholder(dtype=tf.int64,
                                     shape=None,
                                     name='labels')

        # for compatibility with dense model
        self.keep_prob = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name='keep_prob')

    def _build_network(self,
                       inputs: tf.Tensor,
                       sparse_layers: list,
                       activation_fn: callable) -> tf.Tensor:
    
        with tf.variable_scope('network'):
    
            net = inputs
    
            self.weight_tensors = []

            bias_initializer = tf.constant_initializer(0.1)

            # block 1 -- outputs 112x112x64
            net, w, b = L.sparse_conv(net, sparse_layers[0], name="conv1_1")
            self.weight_tensors.append(w)
            net, w, b = L.sparse_conv(net, sparse_layers[1], name="conv1_2")
            self.weight_tensors.append(w)
            net = L.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

            # block 2 -- outputs 56x56x128
            net, w, b = L.sparse_conv(net, sparse_layers[2], name="conv2_1")
            self.weight_tensors.append(w)
            net, w, b = L.sparse_conv(net, sparse_layers[3], name="conv2_2")
            self.weight_tensors.append(w)
            net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

            # # block 3 -- outputs 28x28x256
            net, w, b = L.sparse_conv(net, sparse_layers[4], name="conv3_1")
            self.weight_tensors.append(w)
            net, w, b = L.sparse_conv(net, sparse_layers[5], name="conv3_2")
            self.weight_tensors.append(w)
            net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

            # block 4 -- outputs 14x14x512
            net, w, b = L.sparse_conv(net, sparse_layers[6], name="conv4_1")
            self.weight_tensors.append(w)
            net, w, b = L.sparse_conv(net, sparse_layers[7], name="conv4_2")
            self.weight_tensors.append(w)
            net, w, b = L.sparse_conv(net, sparse_layers[8], name="conv4_3")
            self.weight_tensors.append(w)
            net = L.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

            # block 5 -- outputs 7x7x512
            net, w, b = L.sparse_conv(net, sparse_layers[9], name="conv5_1")
            self.weight_tensors.append(w)
            net, w, b = L.sparse_conv(net, sparse_layers[10], name="conv5_2")
            self.weight_tensors.append(w)
            net, w, b = L.sparse_conv(net, sparse_layers[11], name="conv5_3")
            self.weight_tensors.append(w)
            net = L.pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

            # flatten
            flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
            net = tf.reshape(net, [-1, flattened_shape], name="flatten")

            # fully connected
            net, w, b = L.sparse_fully_connected(net, sparse_layers[12], name="fc6")
            self.weight_tensors.append(w)
            net, w, b = L.sparse_fully_connected(net, sparse_layers[13], name="fc7")
            self.weight_tensors.append(w)
            net, w, b = L.sparse_fully_connected(net, sparse_layers[14], name="fc8_2", activation_fn=None)
            self.weight_tensors.append(w)

            return net

    def _create_loss(self,
                     logits: tf.Tensor,
                     labels: tf.Tensor) -> tf.Tensor:
    
        with tf.variable_scope('loss'):
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=logits, labels=labels,
                                            name='classification_loss')
    
            classification_loss = tf.reduce_mean(classification_loss,
                                                 name='classification_loss_averaged')

            return classification_loss
