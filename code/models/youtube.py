
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from . import layers
from .base import BaseModel

from config import hparams as FLAGS


class YoutubeModelLogistic(BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):
    """Creates a  logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    self.is_training = is_training
    config = FLAGS.youtube

    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      regularizer = None
    else:
      regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    activation = model_input

    with tf.variable_scope('dense'):
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][0] or feature_size
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


class YoutubeModelCirculant(BaseModel):

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

    activation = model_input

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
        activation = tf.nn.leaky_relu(activation, 0.5)
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


class YoutubeModelCirculantWithDense(BaseModel):

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

    activation = model_input

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
        activation = tf.nn.leaky_relu(activation, 0.5)
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


class YoutubeModelTensorTrainWithDense(BaseModel):

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
      # regularizer = reg_fn(l=FLAGS.weight_decay_rate)
      # scale = tf.constant(FLAGS.weight_decay_rate)
      # regularizer = t3f.l2_regularizer(scale)
      regularizer = None

    slope = config['leaky_slope']
    rank = config['rank']
    activation = tf.layers.flatten(model_input)

    tt_shape = [(18, 4, 4, 4), (18, 4, 4, 4)]

    for i in range(config["n_layers"]):
      with tf.variable_scope("tensor_train{}".format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config["hidden"][i] or feature_size
        bias_initializer = tf.random_normal_initializer(stddev=0.01)
        cls_layer = layers.TensorTrainLayer(rank, tt_shape, num_hidden,
                                            bias_initializer=bias_initializer,
                                            use_bias=config["use_bias"],
                                            regularizer=regularizer)
        activation = cls_layer.matmul(activation)
        activation = tf.nn.leaky_relu(activation, slope)
        self._activation_summary(activation)

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


class YoutubeModelTensorTrain(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.tensor_train
    if type(config['hidden']) == int:
      config['hidden'] = [config['hidden']] * config['n_layers']
    assert config["n_layers"] == len(config["hidden"])

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      regularizer = None
    else:
      # regularizer = reg_fn(l=FLAGS.weight_decay_rate)
      # scale = tf.constant(FLAGS.weight_decay_rate)
      # regularizer = t3f.l2_regularizer(scale)
      regularizer = None

    rank = config['rank']
    activation = tf.layers.flatten(model_input)

    tt_shape = [(18, 4, 4, 4), (18, 4, 4, 4)]

    for i in range(config["n_layers"]):
      with tf.variable_scope("tensor_train{}".format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config["hidden"][i] or feature_size
        bias_initializer = tf.random_normal_initializer(stddev=0.01)
        cls_layer = layers.TensorTrainLayer(rank, tt_shape, num_hidden,
                                            bias_initializer=bias_initializer,
                                            use_bias=config["use_bias"],
                                            regularizer=regularizer)
        activation = cls_layer.matmul(activation)
        activation = tf.nn.relu(activation)
        self._activation_summary(activation)

    tt_shape = [(18, 4, 4, 4), (16, 16, 4, 4)]

    # classification layer
    with tf.variable_scope("classification"):
      bias_initializer = tf.random_normal_initializer(stddev=0.01)
      cls_layer = layers.TensorTrainLayer(rank, tt_shape, 16*16*4*4,
                                          bias_initializer=bias_initializer,
                                          use_bias=config["use_bias"],
                                          regularizer=regularizer)
      activation = cls_layer.matmul(activation)
      self._activation_summary(activation)
      activation = activation[..., :n_classes]
    return activation


class YoutubeModelLowRank(BaseModel):

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
        activation = tf.nn.relu(activation)
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
