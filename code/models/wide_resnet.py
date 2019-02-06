
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from .base import BaseModel
from config import hparams as FLAGS

class WideResnetModel(BaseModel):
  """Wide ResNet model.
     https://arxiv.org/abs/1605.07146
  """

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):

    assert(strides[1] == strides[2])
    stride = strides[1]

    # with tf.variable_scope(name):
    #   n = filter_size * filter_size * out_filters
    #   shape = [filter_size, filter_size, in_filters, out_filters]
    #   kernel = tf.get_variable('kernel', shape,
    #     initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)),
    #     regularizer=self.regularizer)
    #   x = tf.nn.conv2d(x, kernel, strides, padding='SAME')

    n = filter_size * filter_size * out_filters
    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/n))
    kernel_size = (filter_size, filter_size)
    x = tf.layers.conv2d(x, out_filters, kernel_size,
                         strides=(stride, stride),
                         padding='same',
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=self.regularizer,
                         use_bias=False,
                         name=name)
    return x

  def _batch_normalization(self, x):
     return tf.layers.batch_normalization(x,
               training=self.is_training,
               beta_regularizer=self.regularizer,
               gamma_regularizer=self.regularizer)

  def _residual(self, x, in_filter, out_filter, stride):
    """Residual unit with 2 sub layers."""

    x_orig = x
    strides = [1, stride, stride, 1]

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, strides)
      x = self._batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.leaky_slope)
    with tf.variable_scope('sub2'):
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
      x = self._batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.leaky_slope)
      if self.dropout > 0:
        x = tf.layers.dropout(x, rate=self.dropout, training=self.is_training)

    if in_filter == out_filter:
      x = x + x_orig
    else:
      x = x + self._conv('conv_orig', x_orig, 1, in_filter, out_filter, strides)

    x = self._batch_normalization(x)
    x = tf.nn.leaky_relu(x, self.leaky_slope)
    return x

  def _unit(self, x, in_filter, out_filter, n, stride, unit_id):
    for i in range(n):
      with tf.variable_scope('group{}_block_{}'.format(unit_id, i)):
        x = self._residual(x, in_filter, out_filter, stride if i == 0 else 1)
    return x

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):
    """Build the core model within the graph."""

    self.config = config = FLAGS.wide_resnet
    self.n_classes = n_classes
    self.is_training = is_training
    self.k = self.config['widen_factor']
    self.depth = self.config['depth']
    self.leaky_slope = config['leaky_slope']
    self.dropout = config['dropout']

    assert(self.depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (self.depth - 4) // 6
    filters = [16, 16*self.k, 32*self.k, 64*self.k]

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      self.regularizer = None
    else:
      self.regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    x = model_input

    with tf.variable_scope('init'):
      filter_size = 3
      in_filters  = x.get_shape()[-1]
      out_filters = 16
      x = self._conv("init_conv", x, filter_size, in_filters, out_filters, [1, 1, 1, 1])
      x = self._batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.leaky_slope)

    x = self._unit(x, filters[0], filters[1], n, 1, 1)
    x = self._unit(x, filters[1], filters[2], n, 2, 2)
    x = self._unit(x, filters[2], filters[3], n, 2, 3)

    with tf.variable_scope('unit_last'):
      x = tf.layers.average_pooling2d(x, [8, 8], [1, 1])
      x = tf.layers.flatten(x)

    with tf.variable_scope('logits') as scope:
      feature_size = x.get_shape().as_list()[-1]
      stddev = 1/np.sqrt(feature_size)
      kernel_initializer = tf.random_normal_initializer(stddev=stddev)
      logits = tf.layers.dense(x, n_classes, use_bias=True,
         kernel_initializer=kernel_initializer,
         kernel_regularizer=self.regularizer,
         bias_regularizer=self.regularizer,
         activation=None)

    return logits
