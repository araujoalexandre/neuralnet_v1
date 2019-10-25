# Copyright 2019 Alexandre Araaujo. All Rights Reserved.
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
# 
# This file has been modified by My Alexandre Araujo.
# ==============================================================================
"""Base model configuration for CNN benchmarks."""

from collections import namedtuple

from absl import logging
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

from . import convnet_builder

# BuildNetworkResult encapsulate the result (e.g. logits) of a
# Model.build_network() call.
BuildNetworkResult = namedtuple(
    'BuildNetworkResult',
    [
        'logits',  # logits of the network
        'extra_info',  # Model specific extra information
    ])


class Model(object):
  """Base model config for DNN benchmarks."""

  def __init__(self,
               model_name,
               params=None):
    self.model_name = model_name

    # use_tf_layers specifies whether to build the model using tf.layers.
    # fp16_vars specifies whether to create the variables in float16.
    if params:
      self.use_tf_layers = params.use_tf_layers
      self.fp16_vars = params.fp16_vars
      self.data_type = tf.float16 if params.use_fp16 else tf.float32
    else:
      self.use_tf_layers = True
      self.fp16_vars = False
      self.data_type = tf.float32

  def get_model_name(self):
    return self.model_name

  def filter_l2_loss_vars(self, variables):
    """Filters out variables that the L2 loss should not be computed for.

    By default, this filters out batch normalization variables and keeps all
    other variables. This behavior can be overridden by subclasses.

    Args:
      variables: A list of the trainable variables.

    Returns:
      A list of variables that the L2 loss should be computed for.
    """
    return [v for v in variables if 'batchnorm' not in v.name]

  def filter_l1_loss_vars(self, variables):
    """Filters out variables that the L1 loss should not be computed for.

    By default, no filter.

    Args:
      variables: A list of the trainable variables.

    Returns:
      A list of variables that the L1 loss should be computed for.
    """
    return variables

  def get_learning_rate(self, global_step, batch_size):
    del global_step
    del batch_size
    return self.learning_rate

  def build_network(self, inputs, phase_train, nclass):
    """Builds the forward pass of the model.

    Args:
      inputs: The list of inputs, including labels
      phase_train: True during training. False during evaluation.
      nclass: Number of classes that the inputs can belong to.

    Returns:
      A BuildNetworkResult which contains the logits and model-specific extra
        information.
    """
    raise NotImplementedError('Must be implemented in derived classes')

  def loss_function(self, inputs, build_network_result):
    """Returns the op to measure the loss of the model.

    Args:
      inputs: the input list of the model.
      build_network_result: a BuildNetworkResult returned by build_network().

    Returns:
      The loss tensor of the model.
    """
    raise NotImplementedError('Must be implemented in derived classes')


  def regularization(self, trainable_variables, rel_device_num, n_devices,
                     params):
    """Return the regularization of the model.

    Args:
      trainable_variables: trainable variables of the model.
      rel_device_num: device id.
      n_devices: the number of devices use for training.
      params: configuration of the training & model.

    Returns:
      The regularization tensor of the model.
    """
    raise NotImplementedError('Must be implemented in derived classes')

  def accuracy_function(self, inputs, logits):
    """Returns the ops to measure the accuracy of the model."""
    raise NotImplementedError('Must be implemented in derived classes')

  def postprocess(self, results):
    """Postprocess results returned from model in Python."""
    return results


class CNNModel(Model):
  """Base model configuration for CNN benchmarks."""

  def __init__(self,
               model,
               params=None):
    super(CNNModel, self).__init__(
        model, params=params)
    if params is not None and params.dataset == 'mnist':
      self.depth = 1
    else:
      self.depth = 3
    self.params = params
    self.data_format = params.data_format if params else 'NCHW'

  def skip_final_affine_layer(self):
    """Returns if the caller of this class should skip the final affine layer.

    Normally, this class adds a final affine layer to the model after calling
    self.add_inference(), to generate the logits. If a subclass override this
    method to return True, the caller should not add the final affine layer.

    This is useful for tests.
    """
    return False

  def add_inference(self, cnn):
    """Adds the core layers of the CNN's forward pass.

    This should build the forward pass layers, except for the initial transpose
    of the images and the final Dense layer producing the logits. The layers
    should be build with the ConvNetBuilder `cnn`, so that when this function
    returns, `cnn.top_layer` and `cnn.top_size` refer to the last layer and the
    number of units of the layer layer, respectively.

    Args:
      cnn: A ConvNetBuilder to build the forward pass layers with.
    """
    del cnn
    raise NotImplementedError('Must be implemented in derived classes')

  def gpu_preprocess_nhwc(self, images, phase_train=True):
    del phase_train
    return images

  def build_network(self,
                    inputs,
                    phase_train=True,
                    nclass=1001):
    """Returns logits from input images.

    Args:
      inputs: The input images and labels
      phase_train: True during training. False during evaluation.
      nclass: Number of classes that the images can belong to.

    Returns:
      A BuildNetworkResult which contains the logits and model-specific extra
        information.
    """
    images = inputs[0]
    images = self.gpu_preprocess_nhwc(images, phase_train)
    if self.data_format == 'NCHW':
      images = tf.transpose(images, [0, 3, 1, 2])
    var_type = tf.float32
    if self.data_type == tf.float16 and self.fp16_vars:
      var_type = tf.float16
    network = convnet_builder.ConvNetBuilder(
        images, self.depth, nclass, phase_train, self.use_tf_layers,
        self.data_format, self.data_type, var_type)
    with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
      self.add_inference(network)
      # Add the final fully-connected class layer
      logits = (
          network.affine(nclass, activation='linear')
          if not self.skip_final_affine_layer() else network.top_layer)
      aux_logits = None
      if network.aux_top_layer is not None:
        with network.switch_to_aux_top_layer():
          aux_logits = network.affine(nclass, activation='linear', stddev=0.001)
    if self.data_type == tf.float16:
      logits = tf.cast(logits, tf.float32)
      if aux_logits is not None:
        aux_logits = tf.cast(aux_logits, tf.float32)
    return BuildNetworkResult(
        logits=logits, extra_info=None if aux_logits is None else aux_logits)

  def loss_function(self, inputs, build_network_result):
    """Returns the op to measure the loss of the model."""
    logits = build_network_result.logits
    _, labels = inputs
    aux_logits = build_network_result.extra_info
    with tf.name_scope('xentropy'):
      cross_entropy = tf_v1.losses.sparse_softmax_cross_entropy(
          logits=logits, labels=labels)
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    if aux_logits is not None:
      with tf.name_scope('aux_xentropy'):
        aux_cross_entropy = tf_v1.losses.sparse_softmax_cross_entropy(
            logits=aux_logits, labels=labels)
        aux_loss = 0.4 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')
        loss = tf.add_n([loss, aux_loss])
    return loss

  def regularization(self, trainable_variables, rel_device_num, n_devices,
                     params):
    """Returns the regularization of the model."""
    l2_loss = tf.constant(0.)
    with tf.name_scope('l2_loss'):
      filtered_params = self.filter_l2_loss_vars(trainable_variables)
      if rel_device_num == n_devices - 1:
        # We compute the L2 loss for only one device instead of all of them,
        # because the L2 loss for each device is the same. To adjust for this,
        # we multiply the L2 loss by the number of devices. We choose the
        # last device because for some reason, on a Volta DGX1, the first four
        # GPUs take slightly longer to complete a step than the last four.
        if params.single_l2_loss_op:
          reshaped_params = [tf.reshape(p, (-1,)) for p in filtered_params]
          l2_loss = tf.nn.l2_loss(tf.concat(reshaped_params, axis=0))
        else:
          l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in filtered_params])
    return n_devices * params.weight_decay * l2_loss

