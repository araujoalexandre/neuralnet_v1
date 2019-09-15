# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trivial model configuration."""
import logging

import numpy as np
import tensorflow as tf

from . import model as model_lib


class TrivialModel(model_lib.CNNModel):
  """Trivial model configuration."""

  def __init__(self, params=None):
    super(TrivialModel, self).__init__(
        'trivial', params=params)

  def add_inference(self, cnn):
    cnn.flatten()
    cnn.affine(32*32)
    cnn.affine(32*32)



class TrivialSSD300Model(model_lib.CNNModel):
  """Trivial SSD300 model configuration."""

  def __init__(self, params=None):
    super(TrivialSSD300Model, self).__init__(
        'trivial', params=params)

  def add_inference(self, cnn):
    cnn.reshape([-1, 300 * 300 * 3])
    cnn.affine(4096)

  def get_input_shapes(self, subset):
    return [[self.batch_size, 300, 300, 3],
            [self.batch_size, 8732, 4],
            [self.batch_size, 8732, 1],
            [self.batch_size]]

  def loss_function(self, inputs, build_network_result):
    images, _, _, labels = inputs
    labels = tf.cast(labels, tf.int32)
    return super(TrivialSSD300Model, self).loss_function(
        (images, labels), build_network_result)
