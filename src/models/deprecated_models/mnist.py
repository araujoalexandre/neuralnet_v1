
import re
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from . import layers

from config import hparams as FLAGS


class MnistModelLeNet:

  def create_model(self, x, n_classes, is_training, *args, **kwargs):

    self.is_training = is_training

    init = tf.truncated_normal_initializer(stddev=0.01)
    reg = tf.keras.regularizers.l2(0.0005)

    x = tf.layers.conv2d(
           inputs=x,
           filters=32,
           kernel_size=[5, 5],
           padding="same",
           activation=tf.nn.relu,
           kernel_initializer=init,
           kernel_regularizer=reg,
           name='conv1')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2)
    x = tf.layers.conv2d(
           inputs=x,
           filters=64,
           kernel_size=[5, 5],
           padding="same",
           activation=tf.nn.relu,
           kernel_initializer=init,
           kernel_regularizer=reg,
           name='conv2')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(
           inputs=x,
           units=1024,
           use_bias=True,
           activation=tf.nn.relu,
           kernel_initializer=init,
           kernel_regularizer=reg,
           name='dense1')
    x = tf.layers.dropout(x, rate=0.4, training=is_training)
    x = tf.layers.dense(x, units=10,
             kernel_initializer=init, kernel_regularizer=reg)
    return x



class MnistModelDense:

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    self.is_training = is_training
    config = FLAGS.dense
    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    activation = tf.layers.flatten(model_input)

    for i in range(config['n_layers']):
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][i] or feature_size
      initializer = tf.random_normal_initializer(stddev=1/np.sqrt(feature_size))
      activation = tf.layers.dense(activation, num_hidden,
                                   use_bias=config['use_bias'],
                                   activation=tf.nn.relu,
                                   kernel_initializer=initializer)
      tf.summary.histogram('Mnist/Dense/Layer{}'.format(i), activation)

    # classification layer
    feature_size = activation.get_shape().as_list()[-1]
    initializer = tf.random_normal_initializer(stddev=1/np.sqrt(feature_size))
    activation = tf.layers.dense(activation, n_classes, activation=None,
                                 use_bias=config['use_bias'],
                                 kernel_initializer=initializer)
    return activation


class MnistModelCirculant:

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    self.is_training = is_training
    config = FLAGS.circulant
    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    alpha = config['alpha']

    activation = tf.layers.flatten(model_input)
    for i in range(config["n_layers"]):
      with tf.variable_scope("circulant{}".format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config["hidden"][i] or feature_size
        kernel_initializer = tf.random_normal_initializer(
          stddev=alpha/np.sqrt(num_hidden))
        bias_initializer = tf.random_normal_initializer(stddev=0.01)
        cls_layer = layers.CirculantLayer(feature_size, num_hidden,
                                          kernel_initializer=kernel_initializer,
                                          bias_initializer=bias_initializer,
                                          use_diag=config["use_diag"],
                                          use_bias=config["use_bias"])
        activation = cls_layer.matmul(activation)
        activation = tf.nn.leaky_relu(activation, 0.5)
        tf.summary.histogram("activation_{}".format(alpha), activation)

    # classification layer
    with tf.variable_scope("classification"):
      feature_size = activation.get_shape().as_list()[-1]
      kernel_initializer = tf.random_normal_initializer(
        stddev=alpha/np.sqrt(n_classes))
      bias_initializer = tf.random_normal_initializer(stddev=0.01)
      cls_layer = layers.CirculantLayer(feature_size, n_classes,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       use_diag=config["use_diag"],
                                       use_bias=config["use_bias"])
      activation = cls_layer.matmul(activation)
      tf.summary.histogram("classification", activation)
    return activation


class MnistModelToeplitz:

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    self.is_training = is_training
    activation = tf.layers.flatten(model_input)
    feature_size = activation.get_shape().as_list()[-1]

    layer1 = layers.ToeplitzLayer(feature_size, feature_size)
    activation = layer1.matmul(activation)
    activation = tf.nn.relu(activation)

    layer2 = layers.ToeplitzLayer(feature_size, n_classes)
    activation = layer2.matmul(activation)

    return activation
