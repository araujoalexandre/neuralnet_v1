
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from . import layers
from .base import BaseModel

from config import hparams as FLAGS


class Cifar10BaseModel(BaseModel):

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


class Cifar10ModelDense(Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.dense

    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      regularizer = None
    else:
      regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    if config['with_conv']:
      activation = self.convolutional_layers(model_input)
    else:
      activation = tf.layers.flatten(model_input)

    for i in range(config['n_layers']):
      with tf.variable_scope('dense{}'.format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config['hidden'][i] or feature_size
        initializer = tf.random_normal_initializer(
          stddev=1/np.sqrt(feature_size))
        bias_initializer = tf.random_normal_initializer(stddev=0.01)
        activation = tf.layers.dense(activation, num_hidden,
                                     use_bias=config['use_bias'],
                                     activation=tf.nn.relu,
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     bias_initializer=bias_initializer,
                                     bias_regularizer=regularizer)
        self._activation_summary(activation)

    # classification layer
    with tf.variable_scope("classification"):
      feature_size = activation.get_shape().as_list()[-1]
      initializer = tf.random_normal_initializer(stddev=1/np.sqrt(n_classes))
      bias_initializer = tf.random_normal_initializer(stddev=0.01)
      activation = tf.layers.dense(activation, n_classes,
                                   use_bias=config['use_bias'],
                                   activation=None,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=regularizer,
                                   bias_initializer=bias_initializer,
                                   bias_regularizer=regularizer)
      self._activation_summary(activation)
    return activation


class Cifar10ModelCirculant(Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.circulant
    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      regularizer = None
    else:
      regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    alpha = config['alpha']

    if config['with_conv']:
      activation = self.convolutional_layers(model_input)
    else:
      activation = tf.layers.flatten(model_input)

    k = 1
    for i in range(config["n_layers"]):
      with tf.variable_scope("circulant{}".format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config["hidden"][i] or feature_size
        kernel_initializer = tf.random_normal_initializer(
          stddev=alpha/np.sqrt(feature_size + num_hidden))
        bias_initializer = tf.random_normal_initializer(stddev=0.01)
        cls_layer = layers.CirculantLayer(feature_size, num_hidden,
                                          kernel_initializer=kernel_initializer,
                                          bias_initializer=bias_initializer,
                                          use_diag=config["use_diag"],
                                          use_bias=config["use_bias"],
                                          regularizer=regularizer)
        activation = cls_layer.matmul(activation)
        if k % config['non_linear'] == 0:
          activation = tf.nn.leaky_relu(activation, config['leaky_slope'])
        k += 1
        self._activation_summary(activation)

    # classification layer
    with tf.name_scope("classification"):
      kernel_initializer = tf.random_normal_initializer(
        stddev=alpha/np.sqrt(feature_size + n_classes))
      bias_initializer = tf.random_normal_initializer(stddev=0.01)
      cls_layer = layers.CirculantLayer(feature_size, n_classes,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       use_diag=config["use_diag"],
                                       use_bias=config["use_bias"],
                                       regularizer=regularizer)
      activation = cls_layer.matmul(activation)
      self._activation_summary(activation)
    return activation


class Cifar10ModelCirculantWithDense(Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.circulant
    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      regularizer = None
    else:
      regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    alpha = config['alpha']

    if config['with_conv']:
      activation = self.convolutional_layers(model_input)
    else:
      activation = tf.layers.flatten(model_input)

    k = 1
    for i in range(config["n_layers"]):
      with tf.variable_scope("circulant{}".format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config["hidden"][i] or feature_size
        kernel_initializer = tf.random_normal_initializer(
          stddev=alpha/np.sqrt(feature_size + num_hidden))
        bias_initializer = tf.random_normal_initializer(stddev=0.01)
        cls_layer = layers.CirculantLayer(feature_size, num_hidden,
                                          kernel_initializer=kernel_initializer,
                                          bias_initializer=bias_initializer,
                                          use_diag=config["use_diag"],
                                          use_bias=config["use_bias"],
                                          regularizer=regularizer)
        activation = cls_layer.matmul(activation)
        if k % config['non_linear'] == 0:
          activation = tf.nn.leaky_relu(activation, config['leaky_slope'])
        k += 1
        self._activation_summary(activation)

    # classification layer
    with tf.variable_scope("classification"):
      feature_size = activation.get_shape().as_list()[-1]
      initializer = tf.random_normal_initializer(stddev=1/np.sqrt(n_classes))
      bias_initializer = tf.random_normal_initializer(stddev=0.01)
      activation = tf.layers.dense(activation, n_classes,
                                   use_bias=config['use_bias'],
                                   activation=None,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=regularizer,
                                   bias_initializer=bias_initializer,
                                   bias_regularizer=regularizer)
      self._activation_summary(activation)
    return activation


class Cifar10ModelToeplitz(Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.toeplitz
    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      regularizer = None
    else:
      regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    alpha = config['alpha']

    if config['with_conv']:
      activation = self.convolutional_layers(model_input)
    else:
      activation = tf.layers.flatten(model_input)

    k = 1
    for i in range(config["n_layers"]):
      with tf.variable_scope("toeplitz{}".format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config["hidden"][i] or feature_size
        bias_initializer = tf.random_normal_initializer(stddev=0.01)
        cls_layer = layers.ToeplitzLayer(feature_size, num_hidden, alpha,
                                         bias_initializer=bias_initializer,
                                         use_bias=config["use_bias"],
                                         regularizer=regularizer)
        activation = cls_layer.matmul(activation)
        if k % config['non_linear'] == 0:
          activation = tf.nn.leaky_relu(activation, config['leaky_slope'])
        k += 1
        self._activation_summary(activation)

    # classification layer
    with tf.variable_scope("classification"):
      bias_initializer = tf.random_normal_initializer(stddev=0.01)
      cls_layer = layers.ToeplitzLayer(feature_size, n_classes, alpha,
                                       bias_initializer=bias_initializer,
                                       use_bias=config["use_bias"],
                                       regularizer=regularizer)
      activation = cls_layer.matmul(activation)
      self._activation_summary(activation)
    return activation


class Cifar10ModelAcdc(Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.acdc
    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      regularizer = None
    else:
      regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    normal_init = config['normal_init']
    alpha = config['alpha']
    slope = config['leaky_slope']
    rand_init, sign_init = config['rand_init'], config['sign_init']

    activation = tf.layers.flatten(model_input)
    activation = activation * 0.1

    for i in range(config["n_layers"]):
      with tf.variable_scope("ACDC{}".format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config["hidden"][i] or feature_size
        start = True if i == 0 else False
        cls_layer = layers.ACDCLayer(feature_size, num_hidden,
                                     start=start,
                                     use_bias=config["use_bias"],
                                     regularizer=regularizer,
                                     rand_init=rand_init,
                                     sign_init=sign_init,
                                     normal_init=normal_init,
                                     alpha=alpha)

        activation = activation / (2*feature_size)
        activation = cls_layer.matmul(activation)
        activation = tf.nn.leaky_relu(activation, slope)
        # p = tf.Variable(tf.random_shuffle(tf.eye(feature_size)),
        #       name='permutation{}'.format(i), trainable=False)
        # activation = tf.matmul(activation, p)
        self._activation_summary(activation)

    # classification layer
    with tf.name_scope("classification"):
      feature_size = activation.get_shape().as_list()[-1]
      cls_layer = layers.ACDCLayer(feature_size, n_classes,
                                   start=False,
                                   use_bias=config["use_bias"],
                                   regularizer=regularizer,
                                   rand_init=rand_init,
                                   sign_init=sign_init)
      # activation = activation / (feature_size)
      activation = cls_layer.matmul(activation)
      self._activation_summary(activation)

    return activation


class Cifar10ModelLowRank(Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.low_rank
    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      regularizer = None
    else:
      regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    rank = config['rank']
    alpha = config['alpha']
    activation = tf.layers.flatten(model_input)

    for i in range(config["n_layers"]):
      with tf.variable_scope("lowrank{}".format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config["hidden"][i] or feature_size
        kernel_initializer = tf.random_normal_initializer(
          stddev=alpha/np.sqrt(num_hidden))
        bias_initializer = tf.random_normal_initializer(stddev=0.01)
        cls_layer = layers.LowRankLayer(rank, feature_size, num_hidden,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        use_bias=config["use_bias"],
                                        regularizer=regularizer)
        activation = cls_layer.matmul(activation)
        activation = tf.nn.leaky_relu(activation, 0.5)
        self._activation_summary(activation)

    # classification layer
    with tf.name_scope("classification"):
      kernel_initializer = tf.random_normal_initializer(
        stddev=alpha/np.sqrt(n_classes))
      bias_initializer = tf.random_normal_initializer(stddev=0.01)
      cls_layer = layers.LowRankLayer(rank, feature_size, n_classes,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      use_bias=config["use_bias"],
                                      regularizer=regularizer)
      activation = cls_layer.matmul(activation)
      self._activation_summary(activation)
    return activation



class Cifar10ModelTensorTrain(Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    import t3f

    config = FLAGS.tensor_train
    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      regularizer = None
    else:
      regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    rank = config['rank']
    tt_shape = [(16, 16, 6, 2), (16, 16, 6, 2)]

    activation = tf.layers.flatten(model_input)

    for i in range(config["n_layers"]):
      with tf.variable_scope("tensor_train{}".format(i), reuse=False):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config["hidden"][i] or feature_size
        bias_initializer = tf.random_normal_initializer(stddev=0.01)
        cls_layer = layers.TensorTrainLayer(rank, tt_shape, num_hidden,
                                            bias_initializer=bias_initializer,
                                            use_bias=config["use_bias"],
                                            regularizer=regularizer)
        activation = cls_layer.matmul(activation)
        # activation = tf.nn.leaky_relu(activation, 0.5)
        activation = tf.nn.relu(activation)
        self._activation_summary(activation)

    tt_shape = [(96, 32), (5, 2)]

    # classification layer
    with tf.variable_scope("classification", reuse=False):
      bias_initializer = tf.random_normal_initializer(stddev=0.01)
      cls_layer = layers.TensorTrainLayer(rank, tt_shape, n_classes,
                                          bias_initializer=bias_initializer,
                                          use_bias=config["use_bias"],
                                          regularizer=regularizer)
      activation = cls_layer.matmul(activation)
      self._activation_summary(activation)
    return activation
