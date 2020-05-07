
import numpy as np
import tensorflow as tf

from . import model as model_lib


class WideResnetModel(model_lib.CNNModel):
  """Wide ResNet model.
     https://arxiv.org/abs/1605.07146
  """
  def __init__(self, params):

    self.params = params
    assert params.data_format == 'NHWC'
    super(WideResnetModel, self).__init__(
      'wide_resnet', params=params)

  def l1_normalize(self, x, dim, epsilon=1e-12, name=None):
    """Normalizes along dimension `dim` using an L1 norm.
    For a 1-D tensor with `dim = 0`, computes
        output = x / max(sum(abs(x)), epsilon)
    For `x` with more dimensions, independently normalizes each 1-D slice along
    dimension `dim`.
    Args:
      x: A `Tensor`.
      dim: Dimension along which to normalize.  A scalar or a vector of
        integers.
      epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
        divisor if `norm < sqrt(epsilon)`.
      name: A name for this operation (optional).
    Returns:
      A `Tensor` with the same shape as `x`.
    """
    with tf.name_scope(name, "l1_normalize", [x]) as name:
      abs_sum = tf.reduce_sum(tf.abs(x), dim, keep_dims = True)
      x_inv_norm = tf.reciprocal(tf.maximum(abs_sum, epsilon))
      return tf.multiply(x, x_inv_norm, name=name)


  def _conv(self, name,  x, filter_size, in_filters, out_filters, strides):
    assert(strides[1] == strides[2])
    stride = strides[1]
    n = filter_size * filter_size * out_filters
    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/n))
    kernel_size = (filter_size, filter_size)
    return tf.layers.conv2d(x, out_filters, kernel_size,
                         strides=(stride, stride),
                         padding='same',
                         kernel_initializer=kernel_initializer,
                         use_bias=False,
                         name=name)


  def _batch_normalization(self, x):
     return tf.layers.batch_normalization(x,
               training=self.is_training)

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

  def add_inference(self, cnn):
    """Build the core model within the graph."""

    self.config = config = self.params.model_params
    self.is_training = cnn.phase_train

    self.k = config['widen_factor']
    self.depth = config['depth']
    self.leaky_slope = config['leaky_slope']
    self.dropout = config['dropout']

    assert(self.depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (self.depth - 4) // 6
    filters = [16, 16*self.k, 32*self.k, 64*self.k]

    x = cnn.top_layer
    with tf.variable_scope('init'):
      filter_size = 3
      in_filters  = x.get_shape()[-1]
      out_filters = 16
      x = self._conv(
        "init_conv", x, filter_size, in_filters, out_filters, [1, 1, 1, 1])
      x = self._batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.leaky_slope)

    x = self._unit(x, filters[0], filters[1], n, 1, 1)
    x = self._unit(x, filters[1], filters[2], n, 2, 2)
    x = self._unit(x, filters[2], filters[3], n, 2, 3)

    with tf.variable_scope('unit_last'):
      x = tf.layers.average_pooling2d(x, [8, 8], [1, 1])
      x = tf.layers.flatten(x)

    cnn.top_size = int(x.get_shape()[-1])
    cnn.top_layer = x
    return x


