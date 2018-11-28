
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

    config = FLAGS.dense
    assert config['n_layers'] == len(config['hidden'])

    activation = tf.layers.flatten(model_input)

    for i in range(config['n_layers']):
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][i] or feature_size
      initializer = tf.random_normal_initializer(stddev=1/np.sqrt(feature_size))
      activation = tf.layers.dense(activation, num_hidden, activation=tf.nn.relu,
                                   kernel_initializer=initializer)
      tf.summary.histogram('Mnist/Dense/Layer{}'.format(i), activation)

    # classification layer
    feature_size = activation.get_shape().as_list()[-1]
    initializer = tf.random_normal_initializer(stddev=1/np.sqrt(feature_size))
    activation = tf.layers.dense(activation, n_classes, activation=None,
                                 kernel_initializer=initializer)
    return activation


class MnistModelGivens(BaseModel):

  def _givens_layers(self, activation, n_givens, shape_in, shape_out=None):
    for i in range(n_givens):
      givens_layer = layers.GivensLayer(shape_in, shape_out=None)
      activation = givens_layer.matmul(activation)
    if shape_out is not None:
      activation = activation[..., :shape_out]
    return activation

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.givens
    assert config['n_layers'] == len(config['hidden'])

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


class Cifar10BaseModel:

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


class Cifar10ModelDense(BaseModel, Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.dense
    activation = self.convolutional_layers(model_input)

    with tf.variable_scope('dense1') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=1/np.sqrt(384))
      activation = tf.layers.dense(activation, config['hidden'][0], use_bias=True,
        kernel_initializer=kernel_initializer, activation=tf.nn.relu)
      self._activation_summary(activation)

    with tf.variable_scope('dense2') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=1/np.sqrt(192))
      activation = tf.layers.dense(activation, config['hidden'][1], use_bias=True,
        kernel_initializer=kernel_initializer, activation=tf.nn.relu)
      self._activation_summary(activation)

    with tf.variable_scope('dense3') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=1/np.sqrt(10))
      activation = tf.layers.dense(activation, num_classes, use_bias=True,
        kernel_initializer=kernel_initializer, activation=None)
      self._activation_summary(activation)

    return activation


class Cifar10ModelGivens(BaseModel, Cifar10BaseModel):

  def _givens_layers(self, activation, n_givens, shape_in, shape_out=None):
    for i in range(n_givens):
      givens_layer = layers.GivensLayer(shape_in, shape_out=None)
      activation = givens_layer.matmul(activation)
    if shape_out is not None:
      activation = activation[..., :shape_out]
    return activation

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.givens

    activation = self.convolutional_layers(model_input)

    assert len(config['hidden']) == 2

    with tf.variable_scope('givens1') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][0] or feature_size
      activation = self._givens_layers(activation, config["n_givens"],
        feature_size, num_hidden)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('givens2') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][1] or feature_size
      activation = self._givens_layers(activation, config["n_givens"],
        feature_size, num_hidden)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('givens3') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      activation = self._givens_layers(activation, config["n_givens"],
        feature_size, n_classes)
      self._activation_summary(activation)

    return activation


class Cifar10ModelCirculant(BaseModel, Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.circulant
    # activation = self.convolutional_layers(model_input)
    activation = tf.layers.flatten(model_input)

    with tf.variable_scope('circulant1') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][0] or feature_size
      layer1 = layers.CirculantLayer(feature_size, num_hidden)
      activation = layer1.matmul(activation)
      bias = tf.get_variable('bias1', shape=(num_hidden, ),
                             initializer=tf.constant_initializer(0.1))
      activation = activation + bias
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('circulant2') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][1] or feature_size
      layer2 = layers.CirculantLayer(feature_size, num_hidden)
      activation = layer2.matmul(activation)
      bias = tf.get_variable('bias2', shape=(num_hidden, ),
                             initializer=tf.constant_initializer(0.1))
      activation = activation + bias
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('circulant3') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer3 = layers.CirculantLayer(feature_size, n_classes)
      activation = layer3.matmul(activation)
      bias = tf.get_variable('bias3', shape=(n_classes, ),
                            initializer=tf.constant_initializer(0.1))
      activation = activation + bias
      self._activation_summary(activation)

    return activation



class Cifar10ModelToeplitz(BaseModel, Cifar10BaseModel):

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

    config = FLAGS.resnet
    print(model_input.get_shape())

    output = Resnet(
      resnet_size=config['resnet_size'],
      bottleneck=config['bottleneck'],
      num_classes=n_classes,
      num_filters=config['num_filters'],
      kernel_size=config['kernel_size'],
      conv_stride=config['conv_stride'],
      first_pool_size=config['first_pool_size'],
      first_pool_stride=config['first_pool_stride'],
      second_pool_size=config['second_pool_size'],
      second_pool_stride=config['second_pool_stride'],
      block_sizes=config['block_sizes'],
      block_strides=config['block_strides'],
      final_size=config['final_size'],
      version=config['version'],
      data_format=config['data_format']
    ).foward(model_input, is_training)
    return output

