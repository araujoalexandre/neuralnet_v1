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

"""Lenet model configuration.

References:
  LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner
  Gradient-based learning applied to document recognition
  Proceedings of the IEEE (1998)
"""

from . import model as model_lib


class Lenet5Model(model_lib.CNNModel):
  """Lenet5."""

  def __init__(self, params=None):
    if params.dataset == 'imagenet':
      assert params.imagenet_image_size == 28
    super(Lenet5Model, self).__init__('lenet5', params=params)

  def add_inference(self, cnn):
    # Note: This matches TF's MNIST tutorial model
    cnn.conv(32, 5, 5)
    cnn.mpool(2, 2)
    cnn.conv(64, 5, 5)
    cnn.mpool(2, 2)
    cnn.flatten()
    cnn.affine(512)


