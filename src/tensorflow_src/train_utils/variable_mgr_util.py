# Copyright 2016 Alexandre Araujo. All Rights Reserved. 
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
"""Utilities for VariableMgr."""
import collections as pycoll
import operator

from absl import logging
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_util


PS_SHADOW_VAR_PREFIX = 'ps_var'

AutoLossScaleParams = pycoll.namedtuple(
    'AutoLossScaleParams',
    [
        # If true, enable automatic loss scaling.
        'enable_auto_loss_scale',
        # The value to scale the loss before computing gradients.
        'loss_scale',
        # Number of normal steps with the current `loss_scale`.
        'loss_scale_normal_steps',
        # Increase loss scale every n steps.
        'inc_loss_scale_every_n',
        # If true, the current worker is chief. The current implementation
        # relies on the chief to update loss_scale value, but in future, we
        # might change this to ask the parameter server to update loss_scales
        # for better performance.
        # TODO(tanmingxing): remove this if loss_scale is updated in ps.
        'is_chief',
    ])


def get_loss_scale_update_op(loss_scale, loss_scale_normal_steps,
                             inc_loss_scale_every_n):
  """Returns the update op for loss scaling variables.

  We maintain the counter `loss_scale_normal_steps` to count the number of steps
  we have been using the current `loss_scale`. In most cases, this function
  increments `loss_scale_normal_steps`. However, if `loss_scale_normal_steps` is
  greater than the threshold `inc_loss_scale_every_n`, we double `loss_scale`
  and reset `loss_scale_normal_steps` to zero.

  This op is only called if the gradients don't have any infs or nans. Instead,
  if infs or nans occur in the gradients, we immeditately halve `loss_scale` and
  reset `loss_scale_normal_steps` to zero.

  Args:
    loss_scale: a tf.Variable represneting the loss_scale value.
    loss_scale_normal_steps: a tf.Variable representing the number of training
      steps that have run since the loss_scale last changed.
    inc_loss_scale_every_n: a Python integer threshold. `loss_scale` is
      increased every `inc_loss_scale_every_n` steps, unless the gradients have
      infs or nans.

  Returns:
    An op for updating `loss_scale` and `loss_scale_normal_steps`.
  """

  def increment_loss_scale_normal_steps_func():
    return tf.group(loss_scale_normal_steps.assign_add(1))

  def increase_loss_scale_func():
    return tf.group(
        tf.assign(loss_scale_normal_steps, 0),
        tf.assign(loss_scale, loss_scale * 2))

  # true_fn and false_fn must have the same type.
  return tf.cond(loss_scale_normal_steps < inc_loss_scale_every_n,
                 increment_loss_scale_normal_steps_func,
                 increase_loss_scale_func)


def append_gradients_with_loss_scale(training_ops, get_apply_gradients_ops_func,
                                     loss_scale_params, grad_has_inf_nan):
  """Selectively appends gradients update ops with loss scaling.

  Args:
    training_ops: a list of training ops to be executed.
    get_apply_gradients_ops_func: a function that returns a list of ops for
      applying gradients. Here, we must pass a function instead of the actual
      list of ops; otherwise, those ops would be executed unconditionally due to
      the semantics of tf.cond.
    loss_scale_params: An AutoLossScaleParams tuple.
    grad_has_inf_nan: Boolean tensor indicating whether the gradients have infs
      or nans.
  """
  is_chief = loss_scale_params.is_chief
  loss_scale = loss_scale_params.loss_scale
  loss_scale_normal_steps = loss_scale_params.loss_scale_normal_steps
  inc_loss_scale_every_n = loss_scale_params.inc_loss_scale_every_n
  enable_auto_loss_scale = loss_scale_params.enable_auto_loss_scale

  if loss_scale is None or not enable_auto_loss_scale or not is_chief:
    training_ops.extend(get_apply_gradients_ops_func())
  else:
    # If nans/infs occurred, skip applying gradients and instead update
    # loss_scale (halve loss_scale and reset loss_scale_normal_steps to zero).
    def update_op_if_nan_or_inf():
      """Update loss_scale and discard gradients if nans/infs occurred."""
      return tf.group(
          tf.assign(loss_scale, loss_scale / 2.),
          tf.assign(loss_scale_normal_steps, 0))

    # Otherwise, apply gradients, and update loss_scale and
    # loss_scale_normal_steps.
    def update_op_if_no_nan_or_inf():
      """Apply gradients, and update loss scaling."""
      return tf.group(
          get_loss_scale_update_op(loss_scale, loss_scale_normal_steps,
                                   inc_loss_scale_every_n),
          *get_apply_gradients_ops_func())

    # TODO(tanmingxing): Add support for independent and distributed all_reduce.
    assert grad_has_inf_nan is not None
    update_op = tf.cond(
        grad_has_inf_nan,
        update_op_if_nan_or_inf,
        update_op_if_no_nan_or_inf,
        name='cond_if_grad_has_inf_nan'
    )
    training_ops.append(update_op)


# To be used with custom_getter on tf.get_variable.
class OverrideCachingDevice(object):
  """Variable getter which caches variables on the least loaded device.

  Variables smaller than a certain threshold are cached on a single specific
  device, as specified in the constructor. All other variables are load balanced
  across a pool of devices, by caching each variable on the least loaded device.

  Note that variable creation only happen when building the model graph on the
  first device (see how it sets the 'reuse' parameter in
  VariableMgr.*.create_outer_variable_scope()). That means, for all other
  devices, the variable scope will reuse the variables created before, which
  requires that we set the caching_device correctly as otherwise it may not be
  able to find the previously created variable and will create a new one. This
  requires when building the model graph on different devices, variables with
  the same name should have same size.
  """

  def __init__(self, devices, device_for_small_variables,
               small_variable_size_threshold):
    self.devices = devices
    self.sizes = [0] * len(self.devices)
    self.device_for_small_variables = device_for_small_variables
    self.small_variable_size_threshold = small_variable_size_threshold

  def __call__(self, getter, *args, **kwargs):
    size = tf.TensorShape(kwargs['shape']).num_elements()
    if size < self.small_variable_size_threshold:
      device_name = self.device_for_small_variables
    else:
      device_index, _ = min(enumerate(self.sizes), key=operator.itemgetter(1))
      device_name = self.devices[device_index]
      self.sizes[device_index] += size

    kwargs['caching_device'] = device_name
    var = getter(*args, **kwargs)
    return var


# To be used with custom_getter on tf.get_variable. Ensures the created variable
# is in LOCAL_VARIABLES and not GLOBAL_VARIBLES collection.
class OverrideToLocalVariableIfNotPsVar(object):

  # args and kwargs come from the custom_getter interface for Tensorflow
  # variables, and matches tf.get_variable's signature, with the addition of
  # 'getter' at the beginning.
  def __call__(self, getter, name, *args, **kwargs):
    if name.startswith(PS_SHADOW_VAR_PREFIX):
      return getter(*args, **kwargs)

    if 'collections' in kwargs:
      collections = kwargs['collections']
    if not collections:
      collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    else:
      collections = collections[:]
    collections.remove(tf.GraphKeys.GLOBAL_VARIABLES)
    collections.append(tf.GraphKeys.LOCAL_VARIABLES)
    kwargs['collections'] = list(collections)
    return getter(name, *args, **kwargs)


class ParamServerDeviceSetter(object):
  """Helper class to assign variables on the least loaded ps-device."""

  def __init__(self, worker_device, ps_devices):
    """Initializer for ParamServerDevicSetter.

    Args:
      worker_device: the device to use for computer ops.
      ps_devices: a list of device to use for Variable ops. Each variable is
      assigned to the least loaded device.
    """
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ['Variable', 'VariableV2']:
      return self.worker_device

    device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name


def aggregate_gradients_using_copy_with_device_selection(
    benchmark_cnn, tower_grads, use_mean, check_inf_nan):
  """Aggregate gradients, controlling device for the aggregation.

  Args:
    benchmark_cnn: benchmark_cnn class.
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over towers. The inner list is over individual gradients.
    use_mean: if True, mean is taken, else sum of gradients is taken.
    check_inf_nan: If true, check grads for nans and infs.

  Returns:
    The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
      gradient has been averaged across all towers. The variable is chosen from
      the first tower. The has_nan_or_inf indicates the grads has nan or inf.
  """
  if benchmark_cnn.local_parameter_device_flag == 'gpu':
    avail_devices = benchmark_cnn.raw_devices
  else:
    avail_devices = [benchmark_cnn.param_server_device]
  agg_grads = []
  has_nan_or_inf_list = []
  for i, single_grads in enumerate(zip(*tower_grads)):
    with tf.device(avail_devices[i % len(avail_devices)]):
      grad_and_var, has_nan_or_inf = aggregate_single_gradient_using_copy(
          single_grads, use_mean, check_inf_nan)
      agg_grads.append(grad_and_var)
      has_nan_or_inf_list.append(has_nan_or_inf)
  if check_inf_nan:
    return agg_grads, tf.reduce_any(has_nan_or_inf_list)
  else:
    return agg_grads, None


def aggregate_gradients_using_copy_with_variable_colocation(
    tower_grads, use_mean, check_inf_nan):
  """Aggregate gradients, colocating computation with the gradient's variable.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over towers. The inner list is over individual gradients. All variables
      of the same gradient across towers must be the same (that is,
      tower_grads[x][a][1] == tower_grads[y][a][1] for all indices x, y, and a)
    use_mean: if True, mean is taken, else sum of gradients is taken.
    check_inf_nan: If true, check grads for nans and infs.

  Returns:
    The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
      gradient has been averaged across all towers. The variable is chosen from
      the first tower. The has_nan_or_inf indicates the grads has nan or inf.
  """
  agg_grads = []
  has_nan_or_inf_list = []
  for single_grads in zip(*tower_grads):
    # Note that each single_grads looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    var = single_grads[0][1]

    for _, v in single_grads:
      assert v == var

    with tf.device(var.device):
      grad_and_var, has_nan_or_inf = aggregate_single_gradient_using_copy(
          single_grads, use_mean, check_inf_nan)
      agg_grads.append(grad_and_var)
      has_nan_or_inf_list.append(has_nan_or_inf)

  if check_inf_nan:
    return agg_grads, tf.reduce_any(has_nan_or_inf_list)
  else:
    return agg_grads, None


def aggregate_gradients_using_copy(tower_grads, use_mean, check_inf_nan):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over towers. The inner list is over individual gradients.
    use_mean: if True, mean is taken, else sum of gradients is taken.
    check_inf_nan: check grads for nans and infs.

  Returns:
    The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
      gradient has been averaged across all towers. The variable is chosen from
      the first tower. The has_nan_or_inf indicates the grads has nan or inf.
  """
  agg_grads = []
  has_nan_or_inf_list = []

  for single_grads in zip(*tower_grads):
    grad_and_var, has_nan_or_inf = aggregate_single_gradient_using_copy(
        single_grads, use_mean, check_inf_nan)
    agg_grads.append(grad_and_var)
    has_nan_or_inf_list.append(has_nan_or_inf)

  if check_inf_nan:
    return agg_grads, tf.reduce_any(has_nan_or_inf_list)
  else:
    return agg_grads, None


def aggregate_single_gradient_using_copy(grad_and_vars, use_mean,
                                         check_inf_nan):
  """Calculate the average gradient for a shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    grad_and_vars: A list or tuple of (gradient, variable) tuples. Each
      (gradient, variable) pair within the outer list represents the gradient
      of the variable calculated for a single tower, and the number of pairs
      equals the number of towers.
    use_mean: if True, mean is taken, else sum of gradients is taken.
    check_inf_nan: check grads for nans and infs.

  Returns:
    The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
      gradient has been averaged across all towers. The variable is chosen from
      the first tower. The has_nan_or_inf indicates the grads has nan or inf.
  """
  grads = [g for g, _ in grad_and_vars]
  if any(isinstance(g, tf.IndexedSlices) for g in grads):
    # TODO(reedwm): All-reduce IndexedSlices more effectively.
    grad = gradients_util._AggregateIndexedSlicesGradients(grads)  # pylint: disable=protected-access
  else:
    grad = tf.add_n(grads)

  if use_mean and len(grads) > 1:
    grad = tf.scalar_mul(1.0 / len(grads), grad)

  v = grad_and_vars[0][1]
  if check_inf_nan:
    with tf.name_scope('check_for_inf_and_nan'):
      has_nan_or_inf = tf.logical_not(tf.reduce_all(tf.is_finite(grads)))
    return (grad, v), has_nan_or_inf
  else:
    return (grad, v), None
