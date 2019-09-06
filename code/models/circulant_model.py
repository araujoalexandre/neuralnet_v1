
import numpy as np
import tensorflow as tf
from tensorflow import logging

from models import model as model_lib


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

