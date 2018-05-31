import tensorflow as tf
tf.set_random_seed(123)
import numpy as np
np.random.seed(123)

# from tensorflow.examples.tutorials.mnist import input_data
import input_data
# mnist = input_data.read_data_sets("MNIST_data/")
data_set = input_data.read_data_sets("train.zip")

train_data_provider = data_set.train
validation_data_provider = data_set.validation
test_data_provider = data_set.test

from networks import network_dense
from configs import ConfigNetworkDense as config

# create a classifier
classifier = network_dense.FullyConnectedClassifier(input_size=config.input_size,
                                                    n_classes=config.n_classes,
                                                    layer_sizes=config.layer_sizes,
                                                    model_path=config.model_path,
                                                    dropout=config.dropout,
                                                    weight_decay=config.weight_decay,
                                                    activation_fn=config.activation_fn)

# than train it
classifier.fit(n_epochs=config.n_epochs,
               batch_size=config.batch_size,
               learning_rate_schedule=config.learning_rate_schedule,
               train_data_provider=train_data_provider,
               validation_data_provider=validation_data_provider,
               test_data_provider=test_data_provider)