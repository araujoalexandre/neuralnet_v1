
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
    bias_initializer=None, kernel_regularizer=None, bias_regularizer=None):

    self.shape_in, self.shape_out = shape_in, shape_out
    size = np.max([shape_in, shape_out])
    stddev = 1/np.sqrt(size)
    self.kernel_initializer = tf.random_normal_initializer(stddev=stddev)
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer

    self.padding = True
    if shape_out < shape_in:
        self.padding = False

    self.size = np.max([shape_in, shape_out])
    shape = (self.size, )
    self.kernel = tf.get_variable(name='kernel', shape=shape,
      initializer=self.kernel_initializer, regularizer=self.kernel_regularizer)

  def matmul(self, input_data):
    padding_size = (np.abs(self.size - self.shape_in))
    paddings = ((0, 0), (padding_size, 0))
    data = tf.pad(input_data, paddings) if self.padding else input_data
    act_fft = tf.spectral.rfft(data)
    kernel_fft = tf.spectral.rfft(self.kernel[::-1])
    ret_mul = tf.multiply(act_fft, kernel_fft)
    ret = tf.spectral.irfft(ret_mul)
    ret = tf.cast(ret, tf.float32)
    ret = tf.manip.roll(ret, 1, axis=1)
    return ret[..., :self.shape_out]
