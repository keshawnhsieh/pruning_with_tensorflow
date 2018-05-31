from typing import Union

import tensorflow as tf
from tqdm import tqdm
import numpy as np
from datetime import datetime

from networks.network_base import BaseNetwork
from utils import tensorflow_utils

import layers as L

class FullyConnectedClassifier(BaseNetwork):

    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 layer_sizes: list,
                 model_path: str,
                 activation_fn=tf.nn.relu,
                 dropout=0.25,
                 momentum=0.9,
                 weight_decay=0.0005,
                 scope='FullyConnectedClassifier',
                 verbose=True,
                 pruning_threshold=None):

        """Create an instance of FullyConnectedClassifier"""

        self.input_size = input_size
        self.n_classes = n_classes
        self.layer_sizes = layer_sizes + [n_classes]
        self.model_path = model_path
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scope = scope
        self.verbose = verbose
        self.pruning_threshold = pruning_threshold

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(self.scope):

                self._create_placeholders()

                self.logits = self._build_network(inputs=self.inputs,
                                                  layer_sizes=self.layer_sizes,
                                                  activation_fn=self.activation_fn,
                                                  keep_prob=self.keep_prob)

                self.loss = self._create_loss(logits=self.logits,
                                              labels=self.labels,
                                              weight_decay=self.weight_decay)

                self.train_op = self._create_optimizer(self.loss,
                                                       learning_rate=self.learning_rate,
                                                       momentum=momentum,
                                                       threshold=pruning_threshold)

                self._create_metrics(logits=self.logits,
                                     labels=self.labels,
                                     loss=self.loss)

                self.saver = self._create_saver(tf.global_variables())
                self.init_variables(tf.global_variables())

                if self.verbose:
                    print('\nSuccessfully created graph for {model}.'.format(
                                                                model=self.scope))
                    print('Number of parameters (four bytes == 1 parameter): {}.\n'.format(
                        int(self.number_of_parameters(tf.trainable_variables()))))


    def _create_placeholders(self):

        # create input nodes of a graph
    
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     # shape=(None, self.input_size),
                                     shape=(None, 224, 224, 3),
                                     name='inputs')
    
        self.labels = tf.placeholder(dtype=tf.int64,
                                     shape=None,
                                     name='labels')
    
        self.keep_prob = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name='keep_prob')
    
        self.learning_rate = tf.placeholder(dtype=tf.float32,
                                            shape=(),
                                            name='learning_rate')
    
    def _build_network(self,
                       inputs: tf.Tensor,
                       layer_sizes: list,
                       activation_fn: callable,
                       keep_prob: Union[tf.Tensor, float]) -> tf.Tensor:

        with tf.variable_scope('network'):
    
            net = inputs
    
            self.weight_matrices = []
            self.biases = []

            weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
            bias_initializer = tf.constant_initializer(0.1)

            # dynamically create a network

            # block 1 -- outputs 112x112x64
            net, w, b = L.conv(net, name="conv1_1", kh=3, kw=3, n_out=64)
            self._add_w_b(w, b)
            net, w, b = L.conv(net, name="conv1_2", kh=3, kw=3, n_out=64)
            self._add_w_b(w, b)
            net = L.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

            # block 2 -- outputs 56x56x128
            net, w, b = L.conv(net, name="conv2_1", kh=3, kw=3, n_out=128)
            self._add_w_b(w, b)
            net, w, b = L.conv(net, name="conv2_2", kh=3, kw=3, n_out=128)
            self._add_w_b(w, b)
            net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

            # # block 3 -- outputs 28x28x256
            net, w, b = L.conv(net, name="conv3_1", kh=3, kw=3, n_out=256)
            self._add_w_b(w, b)
            net, w, b = L.conv(net, name="conv3_2", kh=3, kw=3, n_out=256)
            self._add_w_b(w, b)
            net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

            # block 4 -- outputs 14x14x512
            net, w, b = L.conv(net, name="conv4_1", kh=3, kw=3, n_out=512)
            self._add_w_b(w, b)
            net, w, b = L.conv(net, name="conv4_2", kh=3, kw=3, n_out=512)
            self._add_w_b(w, b)
            net, w, b = L.conv(net, name="conv4_3", kh=3, kw=3, n_out=512)
            self._add_w_b(w, b)
            net = L.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

            # block 5 -- outputs 7x7x512
            net, w, b = L.conv(net, name="conv5_1", kh=3, kw=3, n_out=512)
            self._add_w_b(w, b)
            net, w, b = L.conv(net, name="conv5_2", kh=3, kw=3, n_out=512)
            self._add_w_b(w, b)
            net, w, b = L.conv(net, name="conv5_3", kh=3, kw=3, n_out=512)
            self._add_w_b(w, b)
            net = L.pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

            # flatten
            flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
            net = tf.reshape(net, [-1, flattened_shape], name="flatten")

            # fully connected
            net, w, b = L.fully_connected(net, name="fc6", n_out=4096)
            self._add_w_b(w, b)
            net = tf.nn.dropout(net, keep_prob)
            net, w, b = L.fully_connected(net, name="fc7", n_out=4096)
            self._add_w_b(w, b)
            net = tf.nn.dropout(net, keep_prob)
            net, w, b = L.fully_connected(net, name="fc8_2", n_out=2, activation_fn=None)
            self._add_w_b(w, b)
            return net
    
    def _create_loss(self,
                     logits: tf.Tensor,
                     labels: tf.Tensor,
                     weight_decay: float) -> tf.Tensor:
    
        with tf.variable_scope('loss'):
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=logits, labels=labels,
                                            name='classification_loss')
    
            classification_loss = tf.reduce_mean(classification_loss,
                                                 name='classification_loss_averaged')

            l2_loss = weight_decay * tf.add_n(tf.losses.get_regularization_losses())
    
            return l2_loss + classification_loss

    def _add_w_b(self, w, b):
        self.weight_matrices.append(w)
        self.biases.append(b)
        # L2 loss
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             tf.reduce_sum(w ** 2))

    def _create_optimizer(self,
                          loss: tf.Tensor,
                          learning_rate: Union[tf.Tensor, float],
                          momentum: Union[tf.Tensor, float],
                          threshold: float) -> tf.Operation:

        if threshold is not None:
            return self._create_optimizer_sparse(loss=loss,
                                                 threshold=threshold,
                                                 learning_rate=learning_rate,
                                                 momentum=momentum)
        with tf.variable_scope('optimizer'):

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   momentum=momentum,
                                                   name='optimizer')
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
            #                                    name='adam_optimizer')

            self.global_step = tf.Variable(0)
            train_op = optimizer.minimize(loss,
                                          global_step=self.global_step,
                                          name='train_op')

            return train_op

    def _apply_prune_on_grads(self,
                              grads_and_vars: list,
                              threshold: float):

        # we need to make gradients correspondent
        # to the pruned weights to be zero

        grads_and_vars_sparse = []

        for grad, var in grads_and_vars:
            if 'weights' in var.name:
                small_weights = tf.greater(threshold, tf.abs(var))
                mask = tf.cast(tf.logical_not(small_weights), tf.float32)
                grad = grad * mask

            grads_and_vars_sparse.append((grad, var))
               
        return grads_and_vars_sparse

    def _create_optimizer_sparse(self,
                                 loss: tf.Tensor,
                                 threshold: float,
                                 learning_rate: Union[tf.Tensor, float],
                                 momentum: Union[tf.Tensor, float]) -> tf.Operation:

        with tf.variable_scope('optimizer'):

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   momentum=momentum,
                                                   name='optimizer')
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
            #                                    name='adam_optimizer')

            self.global_step = tf.Variable(0)
            grads_and_vars = optimizer.compute_gradients(loss)
            grads_and_vars_sparse = self._apply_prune_on_grads(grads_and_vars,
                                                               threshold)
            train_op = optimizer.apply_gradients(grads_and_vars_sparse,
                                                 global_step=self.global_step,
                                                 name='train_op')

            return train_op

    def _create_metrics(self,
                        logits: tf.Tensor,
                        labels: tf.Tensor,
                        loss: tf.Tensor):

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _create_saver(self, var_list):

        saver = tf.train.Saver(var_list=var_list)
        return saver

    def fit(self,
            n_epochs: int,
            batch_size: int,
            learning_rate_schedule: callable,
            train_data_provider,
            validation_data_provider,
            test_data_provider):

        n_iterations = train_data_provider.num_examples // batch_size

        for epoch in range(n_epochs):
            print('{time}: Starting epoch {epoch}.\n'.format(time=datetime.now(), epoch=epoch+1))
            for iteration in tqdm(range(n_iterations), ncols=75):

                images, labels = train_data_provider.next_batch(batch_size)

                feed_dict = {self.inputs: images,
                             self.labels: labels,
                             self.learning_rate: learning_rate_schedule(epoch+1),
                             self.keep_prob: 1 - self.dropout} 

                self.sess.run(self.train_op, feed_dict=feed_dict)

                # if iteration % 100 == 0:
                #     loss_val, acc_val = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                #
                #     print('{time}: Step {step} completed.'.format(time=datetime.now(), step=iteration))
                #     print('Accuracy on train batch: {accuracy}, loss on train batch: {loss}'.format(
                #         accuracy=acc_val, loss=loss_val
                #     ))
    
            # evaluate metrics after every epoch
            train_accuracy, train_loss = self.evaluate(train_data_provider,
                                                       batch_size=batch_size)
            validation_accuracy, validation_loss = self.evaluate(validation_data_provider,
                                                                 batch_size=batch_size)

            print('\n{time}: Epoch {epoch} completed.'.format(time=datetime.now(), epoch=epoch+1))
            print('Accuracy on train: {accuracy}, loss on train: {loss}'.format(
                                    accuracy=train_accuracy, loss=train_loss))
            print('Accuracy on validation: {accuracy}, loss on validation: {loss}'.format(
                                    accuracy=validation_accuracy, loss=validation_loss))

            self.save_model(global_step=self.global_step)

        test_accuracy, test_loss = self.evaluate(test_data_provider,
                                                 batch_size=batch_size)

        print('\nOptimization finished.'.format(epoch=epoch+1))
        print('Accuracy on test: {accuracy}, loss on test: {loss}'.format(
                                accuracy=test_accuracy, loss=test_loss))

        # self.save_model(global_step=self.global_step)

    def evaluate(self, data_provider, batch_size: int):

        fetches = [self.accuracy, self.loss]

        n_iterations = data_provider.num_examples // batch_size

        average_accuracy = 0
        average_loss = 0

        for iteration in range(n_iterations):

            images, labels = data_provider.next_batch(batch_size)

            feed_dict = {self.inputs: images,
                         self.labels: labels,
                         self.keep_prob: 1.0} 

            accuracy, loss = self.sess.run(fetches, feed_dict=feed_dict)
            
            average_accuracy += accuracy / n_iterations
            average_loss += loss / n_iterations

        return average_accuracy, average_loss
