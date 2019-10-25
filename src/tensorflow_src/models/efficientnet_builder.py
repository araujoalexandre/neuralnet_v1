# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Model Builder for EfficientNet."""

import functools
import os
import re
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from . import model as model_lib
from . import efficientnet_model



class BlockDecoder(object):
  """Block Decoder for readability."""

  def _decode_block_string(self, block_string):
    """Gets a block through a string notation of arguments."""
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    if 's' not in options or len(options['s']) != 2:
      raise ValueError('Strides options should be a pair of integers.')

    return efficientnet_model.BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]),
                 int(options['s'][1])],
        conv_type=int(options['c']) if 'c' in options else 0,
        fused_conv=int(options['f']) if 'f' in options else 0,
        super_pixel=int(options['p']) if 'p' in options else 0,
        condconv=('cc' in block_string))

  def _encode_block_string(self, block):
    """Encodes a block to a string."""
    args = [
        'r%d' % block.num_repeat,
        'k%d' % block.kernel_size,
        's%d%d' % (block.strides[0], block.strides[1]),
        'e%s' % block.expand_ratio,
        'i%d' % block.input_filters,
        'o%d' % block.output_filters,
        'c%d' % block.conv_type,
        'f%d' % block.fused_conv,
        'p%d' % block.super_pixel,
    ]
    if block.se_ratio > 0 and block.se_ratio <= 1:
      args.append('se%s' % block.se_ratio)
    if block.id_skip is False:  # pylint: disable=g-bool-id-comparison
      args.append('noskip')
    if block.condconv:
      args.append('cc')
    return '_'.join(args)

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.

    Args:
      string_list: a list of strings, each string is a notation of block.

    Returns:
      A list of namedtuples to represent blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args

  def encode(self, blocks_args):
    """Encodes a list of Blocks to a list of strings.

    Args:
      blocks_args: A list of namedtuples to represent blocks arguments.
    Returns:
      a list of strings, each string is a notation of block.
    """
    block_strings = []
    for block in blocks_args:
      block_strings.append(self._encode_block_string(block))
    return block_strings


# def swish(features, use_native=True, use_hard=False):
#   """Computes the Swish activation function.
#
#   We provide three alternnatives:
#     - Native tf.nn.swish, use less memory during training than compsible swish.
#     - Quantization friendly hard swish.
#     - A composible swish, equavilant to tf.nn.swish, but more general for
#       finetuning and TF-Hub.
#
#   Args:
#     features: A `Tensor` representing preactivation values.
#     use_native: Whether to use the native swish from tf.nn that uses a custom
#       gradient to reduce memory usage, or to use customized swish that uses
#       default TensorFlow gradient computation.
#     use_hard: Whether to use quantization-friendly hard swish.
#
#   Returns:
#     The activation value.
#   """
#   if use_native:
#     return tf.nn.swish(features)
#
#   if use_hard:
#     return features * tf.nn.relu6(features + np.float32(3)) * (1. / 6.)
#
#   features = tf.convert_to_tensor(features, name='features')
#   return features * tf.nn.sigmoid(features)


# def efficientnet(width_coefficient=None,
#                  depth_coefficient=None,
#                  dropout_rate=0.2,
#                  drop_connect_rate=0.2):
#   """Creates a efficientnet model."""


# def get_model_params(model_name, override_params):
#   """Get the block args and global params for a given model."""
#   if model_name.startswith('efficientnet'):
#   else:
#     raise NotImplementedError('model name is not pre-defined: %s' % model_name)
#
#   if override_params:
#     # ValueError will be raised here if override_params has fields not included
#     # in global_params.
#     global_params = global_params._replace(**override_params)
#
#   logging.info('global_params= %s', global_params)
#   logging.info('blocks_args= %s', blocks_args)
#   return blocks_args, global_params


# def build_model(images,
#                 model_name,
#                 training,
#                 override_params=None,
#                 model_dir=None,
#                 fine_tuning=False,
#                 features_only=False,
#                 pooled_features_only=False):
#   """A helper functiion to creates a model and returns predicted logits.
#
#   Args:
#     images: input images tensor.
#     model_name: string, the predefined model name.
#     training: boolean, whether the model is constructed for training.
#     override_params: A dictionary of params for overriding. Fields must exist in
#       efficientnet_model.GlobalParams.
#     fine_tuning: boolean, whether the model is used for finetuning.
#     features_only: build the base feature network only (excluding final
#       1x1 conv layer, global pooling, dropout and fc head).
#     pooled_features_only: build the base network for features extraction (after
#       1x1 conv layer and global pooling, but before dropout and fc head).
#
#   Returns:
#     logits: the logits tensor of classes.
#     endpoints: the endpoints for each layer.
#
#   Raises:
#     When model_name specified an undefined model, raises NotImplementedError.
#     When override_params has invalid fields, raises ValueError.
#   """
#   assert isinstance(images, tf.Tensor)
#   assert not (features_only and pooled_features_only)
#   # if not training or fine_tuning:
#   #   if not override_params:
#   #     override_params = {}
#   #   override_params['batch_norm'] = utils.BatchNormalization
#   #   override_params['relu_fn'] = functools.partial(swish, use_native=False)
#   blocks_args, global_params = get_model_params(model_name, override_params)
#
#   with tf.variable_scope(model_name):
#     model = efficientnet_model.Model(blocks_args, global_params)
#     outputs = model(
#         images,
#         training=training,
#         features_only=features_only,
#         pooled_features_only=pooled_features_only)
#   if features_only:
#     outputs = tf.identity(outputs, 'features')
#   elif pooled_features_only:
#     outputs = tf.identity(outputs, 'pooled_features')
#   else:
#     outputs = tf.identity(outputs, 'logits')
#   return outputs, model.endpoints


class EfficientNetModel(model_lib.CNNModel):

  def __init__(self, model_name, params=None):

    super(EfficientNetModel, self).__init__(
      'EfficientNetModel', params=params)

    self.model_name = model_name
    base_config = {
        # (width_coef, depth_coef, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }[model_name]
    (width_coef, depth_coef, image_size, dropout_rate) = base_config 
    if image_size != params.imagenet_image_size:
      raise ValueError(
        "With model '{}' image size should be set to {}".format(
          model_name, image_size))

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    self.global_params = efficientnet_model.GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=0.2,
        data_format='channels_last',
        num_classes=1001,
        width_coefficient=width_coef,
        depth_coefficient=depth_coef,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.nn.swish,
        batch_norm=tf.layers.BatchNormalization,
        use_se=True)
    decoder = BlockDecoder()
    self.blocks_args = decoder.decode(blocks_args)

  def skip_final_affine_layer(self):
    return True

  def add_inference(self, cnn):

    with tf.variable_scope(self.model_name):
      model = efficientnet_model.Model(self.blocks_args, self.global_params)
      outputs = model(cnn.top_layer, training=cnn.phase_train)
      cnn.top_layer = outputs


def create_efficientnet_b0(*args, **kwargs):
  return EfficientNetModel('efficientnet-b0', *args, **kwargs)

def create_efficientnet_b1(*args, **kwargs):
  return EfficientNetModel('efficientnet-b1', *args, **kwargs)

def create_efficientnet_b2(*args, **kwargs):
  return EfficientNetModel('efficientnet-b2', *args, **kwargs)

def create_efficientnet_b3(*args, **kwargs):
  return EfficientNetModel('efficientnet-b3', *args, **kwargs)

def create_efficientnet_b4(*args, **kwargs):
  return EfficientNetModel('efficientnet-b4', *args, **kwargs)

def create_efficientnet_b5(*args, **kwargs):
  return EfficientNetModel('efficientnet-b5', *args, **kwargs)

def create_efficientnet_b6(*args, **kwargs):
  return EfficientNetModel('efficientnet-b6', *args, **kwargs)

def create_efficientnet_b7(*args, **kwargs):
  return EfficientNetModel('efficientnet-b7', *args, **kwargs)


