
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from config import hparams as FLAGS

class GivensLayer:

  def __init__(self, shape_in):
    i, j = 2, 1
    while i >= j:
      i, j = np.random.randint(0, shape_in, size=2)
    index = np.zeros((shape_in, 2))
    index[i, 0], index[j, 1] = 1, 1
    self.index = tf.convert_to_tensor(np.float32(index))
    eye_mask = np.invert(index.sum(axis=1).astype(bool)).astype(np.float32)
    self.eye = tf.diag(eye_mask)
    self.theta = tf.Variable(tf.constant(0.5))

  def matmul(self, input_data):
    theta, index, eye = self.theta, self.index, self.eye
    cos, sin = tf.cos(theta), tf.sin(theta)
    rotation = tf.stack([[cos, -sin], [sin, cos]])
    givens = tf.matmul(tf.matmul(index, rotation), tf.transpose(index))
    givens = givens + eye
    ret = tf.matmul(input_data, givens)
    return ret


class GivensLayer_v2:

  def __init__(self, shape_in, shape_out, n_givens=1):

    self.shape_in = shape_in
    self.shape_out = shape_out
    assert self.shape_in >= self.shape_out

    self.n_givens = n_givens
    self.ij = [self._draw_ij(self.shape_in) for _ in range(n_givens)]
    self.thetas = [self._get_weights('theta{}'.format(i))
                    for i in range(len(self.ij))]

  def _draw_ij(self, size):
    i, j = 1, 0
    while i >= j:
      i, j = np.random.randint(0, size, size=2)
    return i, j

  def _get_weights(self, name=None):
    initializer = tf.random_uniform_initializer()
    return tf.get_variable(
                name=name,
                shape=(1),
                dtype=tf.float32,
                initializer=initializer,
                trainable=True)

  def matmul(self, input_data):
    x = tf.unstack(input_data, axis=1)
    for (i, j), theta in zip(self.ij, self.thetas):
      # print_op = tf.print(theta)
      # with tf.control_dependencies([print_op]):
      #   cost, sint = tf.cos(theta), tf.sin(theta)
      cost, sint = tf.cos(theta), tf.sin(theta)
      xi_new = (x[i] * cost + x[j] * -sint)
      xj_new = (x[i] * sint + x[j] * cost)
      with tf.control_dependencies([xi_new, xj_new]):
        x[i] = tf.squeeze(xi_new)
        x[j] = tf.squeeze(xj_new)
    return tf.stack(x, axis=1)[..., :self.shape_out]


class ToeplitzLayer:

  def __init__(self, shape_in, shape_out, kernel_initializer=None,
    bias_initializer=None, kernel_regularizer=None, bias_regularizer=None):

    self.shape_in, self.shape_out = shape_in, shape_out
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer

    self.padding = True
    if shape_out < shape_in:
        self.padding = False

    self.size = np.max([shape_in, shape_out])
    scale = 1/np.sqrt(self.size)
    self.k1 = np.random.normal(scale=scale, size=self.size).tolist()
    self.k2 = np.random.normal(scale=scale, size=self.size).tolist()
    circ = [0] + list(np.roll(self.k2, -1)[::-1][1:]) + self.k1
    kernel = tf.convert_to_tensor(np.float32(circ))

    self.kernel = tf.get_variable(name='kernel',
      initializer=kernel, regularizer=self.kernel_regularizer)

  def matmul(self, input_data):
    padding_size = (np.abs(self.size*2 - self.shape_in))
    paddings = ((0, 0), (padding_size, 0))
    data = tf.pad(input_data, paddings)
    act_fft = tf.spectral.rfft(data)
    kernel_fft = tf.spectral.rfft(self.kernel[::-1])
    ret_mul = tf.multiply(act_fft, kernel_fft)
    ret = tf.spectral.irfft(ret_mul)
    ret = tf.cast(ret, tf.float32)
    ret = tf.manip.roll(ret, 1, axis=1)
    return ret[..., :self.shape_out]


class CirculantLayer:

  def __init__(self, shape_in, shape_out, kernel_initializer=None,
    diag_initializer=None, bias_initializer=None, regularizer=None,
    use_diag=True, use_bias=True):

    self.use_diag, self.use_bias = use_diag, use_bias
    self.shape_in, self.shape_out = shape_in, shape_out
    size = np.max([shape_in, shape_out])

    self.size = np.max([shape_in, shape_out])
    shape = (self.size, )

    if kernel_initializer is None:
      kernel_initializer = np.random.normal(0, 0.001, size=shape)
      kernel_initializer[0] = 1 + np.random.normal(0, 0.01)
      kernel_initializer = np.float32(kernel_initializer)
      self.kernel = tf.get_variable(name='kernel',
        initializer=kernel_initializer, regularizer=regularizer)
    else:
      self.kernel = tf.get_variable(name='kernel', shape=shape,
        initializer=kernel_initializer, regularizer=regularizer)

    if diag_initializer is None:
      stddev = 1/np.sqrt(size)
      diag_initializer = tf.random_normal_initializer(stddev=stddev)

    if bias_initializer is None:
      bias_initializer = tf.constant_initializer(0.1)

    self.padding = True
    if shape_out < shape_in:
        self.padding = False

    if use_diag:
      self.diag = tf.get_variable(name='diag', shape=shape,
        initializer=diag_initializer, regularizer=regularizer)
    if use_bias:
      self.bias = tf.get_variable(name="bias", shape=[shape_out],
        initializer=bias_initializer, regularizer=regularizer)


  def matmul(self, input_data):
    padding_size = (np.abs(self.size - self.shape_in))
    paddings = ((0, 0), (padding_size, 0))
    data = tf.pad(input_data, paddings) if self.padding else input_data
    if self.use_diag:
      data = np.multiply(data, self.diag)
    act_fft = tf.spectral.rfft(data)
    kernel_fft = tf.spectral.rfft(self.kernel[::-1])
    ret_mul = tf.multiply(act_fft, kernel_fft)
    ret = tf.spectral.irfft(ret_mul)
    ret = tf.cast(ret, tf.float32)
    ret = tf.manip.roll(ret, 1, axis=1)
    ret = ret[..., :self.shape_out]
    if self.use_bias:
      ret = ret + self.bias
    return ret


