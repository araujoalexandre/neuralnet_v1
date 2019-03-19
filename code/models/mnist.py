
import re
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from . import layers
from .base import BaseModel

from config import hparams as FLAGS



class MnistModelDense(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    self.is_training = is_training
    config = FLAGS.dense
    assert config['n_layers'] == len(config['hidden'])

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


class MnistModelGivens(BaseModel):

  def _givens_layers(self, activation, n_givens, shape_in, shape_out=None):
    for i in range(n_givens):
      givens_layer = layers.GivensLayer(shape_in)
      activation = givens_layer.matmul(activation)
    if shape_out is not None:
      activation = activation[..., :shape_out]
    return activation

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    self.is_training = is_training
    config = FLAGS.givens
    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    activation = tf.layers.flatten(model_input)

    for i in range(config['n_layers']):
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][i] or feature_size
      activation = self._givens_layers(activation, config['n_givens'],
        feature_size, num_hidden)
      activation = tf.nn.relu(activation)
      tf.summary.histogram('Mnist/Givens/Layer{}'.format(i), activation)

    # classification layer
    activation = self._givens_layers(activation, config['n_givens'],
      feature_size, n_classes)

    return activation


class MnistModelGivens_v2(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    self.is_training = is_training
    config = FLAGS.givens
    assert config['n_layers'] == len(config['hidden'])

    activation = tf.layers.flatten(model_input)

    for i in range(config['n_layers']):
      with tf.variable_scope('givens{}'.format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config['hidden'][i] or feature_size
        cls_layer = layers.GivensLayer_v2(feature_size, num_hidden,
                                       config['n_givens'])
        activation = cls_layer.matmul(activation)
        activation = tf.nn.tanh(activation)
        tf.summary.histogram('Mnist/Givens/Layer{}'.format(i), activation)

    # classification layer
    cls_layer = layers.GivensLayer_v2(feature_size, n_classes,
                                     config['n_givens'])
    activation = cls_layer.matmul(activation)

    return activation


class MnistModelCirculant(BaseModel):

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


class MnistModelToeplitz(BaseModel):

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
