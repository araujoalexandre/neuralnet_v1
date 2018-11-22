
import re
import numpy as np
import tensorflow as tf
from tensorflow import flags

import layers
from resnet import Resnet

from config import hparams as FLAGS



class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):
    raise NotImplementedError()



################################################################################
#####                             MNIST MODEL                              #####
################################################################################

class MnistModelDense(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    activation = tf.layers.flatten(model_input)
    feature_size = activation.get_shape().as_list()[-1]

    initializer = tf.random_normal_initializer(stddev=1/np.sqrt(feature_size))
    activation = tf.layers.dense(activation, n_classes, activation=tf.nn.relu,
      kernel_initializer=initializer)
    return activation


class MnistModelGivens(BaseModel):

  def _givens_layers(self, model_input, n_givens, shape_in, shape_out=None):
    for i in range(n_givens):
      givens_layer = layers.GivensLayer(shape_in, shape_out)
      activation = givens_layer.matmul(model_input)
      return activation

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    n_givens = FLAGS.givens['n_givens']

    activation = tf.layers.flatten(model_input)
    feature_size = activation.get_shape().as_list()[-1]

    activation = self._givens_layers(model_input, n_givens,
      feature_size, None)
    activation = tf.nn.relu(activation)

    activation = self._givens_layers(model_input, n_givens,
      feature_size, None)

    return activation


class MnistModelCirculant(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    activation = tf.layers.flatten(model_input)
    feature_size = activation.get_shape().as_list()[-1]

    layer1 = layers.CirculantLayer(feature_size, feature_size)
    activation = layer1.matmul(activation)
    activation = tf.nn.relu(activation)

    layer2 = layers.CirculantLayer(feature_size, n_classes)
    activation = layer2.matmul(activation)

    return activation


class MnistModelToeplitz(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    activation = tf.layers.flatten(model_input)
    feature_size = activation.get_shape().as_list()[-1]

    layer1 = layers.ToeplitzLayer(feature_size, feature_size)
    activation = layer1.matmul(activation)
    activation = tf.nn.relu(activation)

    layer2 = layers.ToeplitzLayer(feature_size, n_classes)
    activation = layer2.matmul(activation)

    return activation


################################################################################
#####                             CIFAR MODEL                              #####
################################################################################


class Cifar10BAseModel:

  def _activation_summary(self, x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('tower_[0-9]*/', '', x.op.name)
    tf.summary.histogram('{}/activations'.format(tensor_name), x)
    tf.summary.scalar('{}/sparsity'.format(tensor_name), tf.nn.zero_fraction(x))

  def convolutional_layers(self, images):
    """Build the CIFAR-10 model convolutional layers.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      activation
    """

    # conv1
    with tf.variable_scope('conv1') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=5e-2)
      biases_initializer = tf.constant_initializer(0.1)
      kernel = tf.get_variable('weights', (5, 5, 3, 64),
                               initializer=kernel_initializer)
      biases = tf.get_variable('biases', [64], initializer=biases_initializer)

      activation = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      activation = tf.nn.bias_add(activation, biases)
      activation = tf.nn.relu(activation, name=scope.name)
      self._activation_summary(activation)

    # pool1
    activation = tf.nn.max_pool(activation,
      ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    # norm1
    activation = tf.nn.lrn(activation,
      4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=5e-2)
      biases_initializer = tf.constant_initializer(0.1)
      kernel = tf.get_variable('weights', (5, 5, 64, 64),
                               initializer=kernel_initializer)
      biases = tf.get_variable('biases', [64], initializer=biases_initializer)

      activation = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
      activation = tf.nn.bias_add(activation, biases)
      activation = tf.nn.relu(activation, name=scope.name)
      self._activation_summary(activation)

    # norm2
    activation = tf.nn.lrn(activation,
      4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2
    activation = tf.nn.max_pool(activation,
      ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    return tf.layers.flatten(activation)


class Cifar10ModelDense(BaseModel, Cifar10BAseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    activation = self.convolutional_layers(model_input)

    with tf.variable_scope('dense1') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=1/np.sqrt(384))
      activation = tf.layers.dense(activation, 384, use_bias=True,
        kernel_initializer=kernel_initializer, activation=tf.nn.relu)
      self._activation_summary(activation)

    with tf.variable_scope('dense2') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=1/np.sqrt(192))
      activation = tf.layers.dense(activation, 192, use_bias=True,
        kernel_initializer=kernel_initializer, activation=tf.nn.relu)
      self._activation_summary(activation)

    with tf.variable_scope('dense3') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=1/np.sqrt(10))
      activation = tf.layers.dense(activation, 10, use_bias=True,
        kernel_initializer=kernel_initializer, activation=None)
      self._activation_summary(activation)

    # activation = tf.Print(activation, [activation])

    return activation


class Cifar10ModelGivens(BaseModel, Cifar10BAseModel):

  def _givens_layers(self, model_input, n_givens, shape_in, shape_out=None):
    for i in range(n_givens):
      givens_layer = layers.GivensLayer(shape_in, shape_out)
      activation = givens_layer.matmul(model_input)
    return activation

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    activation = self.convolutional_layers(model_input)

    with tf.variable_scope('givens1') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      activation = self._givens_layers(activation, FLAGS.n_givens,
        feature_size, 384)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('givens2') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      activation = self._givens_layers(activation, FLAGS.n_givens,
        feature_size, 192)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('givens3') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      activation = self._givens_layers(activation, FLAGS.n_givens,
        feature_size, 10)
      self._activation_summary(activation)

    return activation


class Cifar10ModelCirculant(BaseModel, Cifar10BAseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    activation = self.convolutional_layers(model_input)

    with tf.variable_scope('circulant1') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer1 = layers.CirculantLayer(feature_size, 384)
      activation = layer1.matmul(activation)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('circulant2') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer2 = layers.CirculantLayer(feature_size, 192)
      activation = layer2.matmul(activation)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('circulant3') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer3 = layers.CirculantLayer(feature_size, 10)
      activation = layer3.matmul(activation)
      self._activation_summary(activation)

    return activation



class Cifar10ModelToeplitz(BaseModel, Cifar10BAseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    activation = self.convolutional_layers(model_input)

    with tf.variable_scope('toeplitz1') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer1 = layers.ToeplitzLayer(feature_size, 384)
      activation = layer1.matmul(activation)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('toeplitz2') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer2 = layers.ToeplitzLayer(feature_size, 192)
      activation = layer2.matmul(activation)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('toeplitz3') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer3 = layers.ToeplitzLayer(feature_size, 10)
      activation = layer3.matmul(activation)
      self._activation_summary(activation)

    return activation




################################################################################
#####                            RESNET MODEL                              #####
################################################################################

class ResnetModel(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):
    output = Resnet(
      resnet_size=18,
      bottleneck=False,
      num_classes=n_classes,
      num_filters=64,
      kernel_size=3,
      conv_stride=1,
      first_pool_size=0,
      first_pool_stride=2,
      second_pool_size=7,
      second_pool_stride=1,
      block_sizes=[2, 2, 2, 2],
      block_strides=[1, 2, 2, 2],
      final_size=512,
      version=2,
      data_format=None).foward(model_input, is_training)
    return output
