
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

from . import model as model_lib
from .scattering_utils import Scattering


class ScatteringHybridCirculantModel(model_lib.CNNModel):

  def __init__(self, params):
    assert params.data_format == "NCHW"
    self.model_params = params.model_params
    super(ScatteringHybridCirculantModel, self).__init__(
      'ScatteringHybridCirculantModel', params=params)

  def skip_final_affine_layer(self):
    return True

  def add_inference(self, cnn):

    n_layers = self.model_params['n_layers']

    M, N = cnn.top_layer.get_shape().as_list()[-2:]
    cnn.top_layer = Scattering(M=M, N=N, J=2)(cnn.top_layer)
    cnn.batch_norm()

    cnn.top_layer = tf.layers.flatten(cnn.top_layer)
    cnn.top_size = cnn.top_layer.get_shape()[-1].value
    for i in range(n_layers):
      cnn.diagonal_circulant(**self.model_params)
    cnn.diagonal_circulant(num_channels_out=cnn.nclass,
                          activation='linear')



class ScatteringHybridDenseModel(model_lib.CNNModel):

  def __init__(self, params):
    assert params.data_format == "NCHW"
    super(ScatteringHybridDenseModel, self).__init__(
      'ScatteringHybridDenseModel', params=params)

  def add_inference(self, cnn):

    M, N = cnn.top_layer.get_shape().as_list()[-2:]
    cnn.top_layer = Scattering(M=M, N=N, J=2)(cnn.top_layer)
    cnn.batch_norm()

    cnn.flatten()
    cnn.affine(cnn.top_size)


