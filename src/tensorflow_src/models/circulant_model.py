
import logging

import numpy as np
import tensorflow as tf

from . import model as model_lib


class DiagonalCirculantModel(model_lib.CNNModel):

  def __init__(self, params):
    self.model_params = params.model_params
    super(DiagonalCirculantModel, self).__init__(
        'DiagonalCirculantModel', params=params)

  def skip_final_affine_layer(self):
    return True

  def add_inference(self, cnn):
    n_layers = self.model_params['n_layers']
    cnn.top_layer = tf.layers.flatten(cnn.top_layer)
    cnn.top_size = cnn.top_layer.get_shape()[-1].value
    for i in range(n_layers):
      cnn.diagonal_circulant(**self.model_params)
    cnn.diagonal_circulant(num_channels_out=cnn.nclass,
                          activation='linear')


class ConvCirculant(model_lib.CNNModel):

  def __init__(self, params):
    assert params.data_format == 'NCHW'
    self.model_params = params.model_params
    super(ConvCirculant, self).__init__(
      'ConvCirculant', params=params)

  def skip_final_affine_layer(self):
    return False

  def _bottleneck(self, cnn, channel, strides=1, residual=True):
    if residual:
      x = cnn.top_layer
    cnn.conv(channel, 1, 1)
    cnn.depth_conv_circ(**self.model_params)
    if strides > 1:
      cnn.mpool(strides, strides)
    cnn.conv(channel, 1, 1, activation='linear')
    if residual and strides == 1:
      cnn.top_layer = cnn.top_layer + x
    logging.info('shape {}'.format(cnn.top_layer.get_shape()))

  def add_inference(self, cnn):

    self._bottleneck(cnn, 32, strides=1, residual=False)

    self._bottleneck(cnn, 16, strides=1, residual=False)
    self._bottleneck(cnn, 16, strides=1)

    self._bottleneck(cnn, 24, strides=2)
    self._bottleneck(cnn, 24, strides=1)
    self._bottleneck(cnn, 24, strides=1)

    self._bottleneck(cnn, 32, strides=2)
    self._bottleneck(cnn, 32, strides=1)
    self._bottleneck(cnn, 32, strides=1)

    self._bottleneck(cnn, 64, strides=2)
    self._bottleneck(cnn, 64, strides=1)
    self._bottleneck(cnn, 64, strides=1)

    self._bottleneck(cnn, 96, strides=1, residual=False)
    self._bottleneck(cnn, 96, strides=1)
    self._bottleneck(cnn, 96, strides=1)

    cnn.flatten()



class RandomConvDiagonalCirculantModel(model_lib.CNNModel):

  def __init__(self, params):
    self.model_params = params.model_params
    super(RandomConvDiagonalCirculantModel, self).__init__(
        'RandomConvDiagonalCirculantModel', params=params)

  def skip_final_affine_layer(self):
    return True

  def add_inference(self, cnn):
    n_layers = self.model_params['n_layers']

    cnn.conv(32, 3, 3, 1, 1, mode='SAME', trainable=False)
    cnn.mpool(2, 2, 2, 2, mode='VALID')
    logging.info(cnn.top_layer.get_shape())

    cnn.conv(64, 3, 3, 1, 1, mode='SAME', trainable=False)
    cnn.mpool(2, 2, 2, 2, mode='VALID')
    logging.info(cnn.top_layer.get_shape())

    cnn.conv(96, 3, 3, 1, 1, mode='SAME', trainable=False)
    cnn.mpool(2, 2, 2, 2, mode='VALID')
    logging.info(cnn.top_layer.get_shape())

    cnn.top_layer = tf.layers.flatten(cnn.top_layer)
    cnn.top_size = cnn.top_layer.get_shape()[-1].value
    for i in range(n_layers):
      cnn.diagonal_circulant(**self.model_params)
    cnn.diagonal_circulant(num_channels_out=cnn.nclass,
                          activation='linear')










