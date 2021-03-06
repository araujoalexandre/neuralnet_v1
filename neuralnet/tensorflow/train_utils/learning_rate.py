
import logging

import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def cyclic_learning_rate(global_step,
                         learning_rate=0.01,
                         max_lr=0.1,
                         step_size=20.,
                         gamma=0.99994,
                         mode='triangular',
                         name=None):
  """
    function taken from: https://goo.gl/x4drQS
    Applies cyclic learning rate (CLR).
    From the paper:
    Smith, Leslie N.
    "Cyclical learning rates for training neural networks." 2017.
    [https://arxiv.org/pdf/1506.01186.pdf]
    This method lets the learning rate cyclically vary between reasonable
    boundary values achieving improved classification accuracy and often in
    fewer iterations. This code varies the learning rate linearly between the
    minimum (learning_rate) and the maximum (max_lr).

    It returns the cyclic learning rate. It is computed as:
       ```python
       cycle = floor( 1 + global_step /
        ( 2 * step_size ) )
      x = abs( global_step / step_size – 2 * cycle + 1 )
      clr = learning_rate +
        ( max_lr – learning_rate ) * max( 0 , 1 - x )
       ```
      Polices:
        'triangular':
          Default, linearly increasing then linearly decreasing the
          learning rate at each cycle.
         'triangular2':
          The same as the triangular policy except the learning
          rate difference is cut in half at the end of each cycle.
          This means the learning rate difference drops after each cycle.
         'exp_range':
          The learning rate varies between the minimum and maximum
          boundaries and each boundary value declines by an exponential
          factor of: gamma^global_step.
       Example: 'triangular2' mode cyclic learning rate.
        '''python
        ...
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=
          clr.cyclic_learning_rate(global_step=global_step, mode='triangular2'))
        train_op = optimizer.minimize(loss_op, global_step=global_step)
        ...
         with tf.Session() as sess:
            sess.run(init)
            for step in range(1, num_steps+1):
              assign_op = global_step.assign(step)
              sess.run(assign_op)
        ...
         '''
       Args:
        global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
          Global step to use for the cyclic computation.  Must not be negative.
        learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate which is the lower bound
          of the cycle (default = 0.1).
        max_lr:  A scalar. The maximum learning rate boundary.
        step_size: A scalar. The number of iterations in half a cycle.
          The paper suggests step_size = 2-8 x training iterations in epoch.
        gamma: constant in 'exp_range' mode:
          gamma**(global_step)
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
        name: String.  Optional name of the operation.  Defaults to
          'CyclicLearningRate'.
       Returns:
        A scalar `Tensor` of the same type as `learning_rate`.  The cyclic
        learning rate.
      Raises:
        ValueError: if `global_step` is not supplied.
      @compatibility(eager)
      When eager execution is enabled, this function returns
      a function which in turn returns the decayed learning
      rate Tensor. This can be useful for changing the learning
      rate value across different invocations of optimizer functions.
      @end_compatibility
  """
  if global_step is None:
    raise ValueError("global_step is required for cyclic_learning_rate.")

  with ops.name_scope(name, "CyclicLearningRate",
    [learning_rate, global_step]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    step_size = math_ops.cast(step_size, dtype)

    def cyclic_lr():
      """Helper to recompute learning rate; most helpful in eager-mode."""

      # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
      double_step = math_ops.multiply(2., step_size)
      global_div_double_step = math_ops.divide(global_step, double_step)
      cycle = math_ops.floor(math_ops.add(1., global_div_double_step))

      # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
      double_cycle = math_ops.multiply(2., cycle)
      global_div_step = math_ops.divide(global_step, step_size)
      tmp = math_ops.subtract(global_div_step, double_cycle)
      x = math_ops.abs(math_ops.add(1., tmp))

      # computing: clr = learning_rate + (max_lr – learning_rate) * max(0, 1 - x)
      a1 = math_ops.maximum(0., math_ops.subtract(1., x))
      a2 = math_ops.subtract(max_lr, learning_rate)
      clr = math_ops.multiply(a1, a2)

      if mode == 'triangular2':
        clr = math_ops.divide(clr, math_ops.cast(math_ops.pow(2, math_ops.cast(
            cycle-1, tf.int32)), tf.float32))
      if mode == 'exp_range':
        clr = math_ops.multiply(math_ops.pow(gamma, global_step), clr)
      return math_ops.add(clr, learning_rate, name=name)

    if not context.executing_eagerly():
      cyclic_lr = cyclic_lr()
    return cyclic_lr



def get_learning_rate(params, global_step, num_step_by_epoch, model):
  """Returns a learning rate tensor based on global_step.

  Args:
    params: Params tuple, typically created by make_params or
      make_params_from_flags.
    global_step: Scalar tensor representing the global step.
    num_examples_per_epoch: The number of examples per epoch.
    model: The model.Model object to obtain the default learning rate from if no
      learning rate is specified.
    batch_size: Number of examples per step

  Returns:
    A scalar float tensor, representing the learning rate. When evaluated, the
    learning rate depends on the current value of global_step.

  Raises:
    ValueError: Invalid or unsupported params.
  """
  lr_strategy = params.lr_strategy
  lr_params = params.lr_params

  with tf.name_scope('learning_rate'):
    if lr_strategy == "default":
      # get learning rate from model class
      learning_rate = model.get_learning_rate(global_step, batch_size)
    elif lr_strategy == 'piecewise_constant':
      boundaries = lr_params['boundaries']
      values = lr_params['values']
      assert (len(boundaries) + 1) == len(values)
      learning_rate = tf_v1.train.piecewise_constant(
        global_step,
        boundaries=boundaries,
        values=values)
    elif params.lr_strategy == 'exponential_decay':
      if lr_params['decay_steps']:
        decay_steps = int(lr_params['decay_steps'])
      elif lr_params['decay_epochs']:
        decay_steps = int(lr_params['decay_epochs'] * num_step_by_epoch)
      learning_rate = tf_v1.train.exponential_decay(
        learning_rate=lr_params['learning_rate'],
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=lr_params['decay_rate'],
        staircase=True)
    elif params.lr_strategy == 'cyclic_lr':
      learning_rate = cyclic_learning_rate(
          global_step,
          learning_rate=lr_params['min_lr'],
          max_lr=lr_params['max_lr'],
          step_size=lr_params['step_size_lr'],
          gamma=lr_params['gamma'],
          mode=lr_params['mode_cyclic_lr'])
    else:
      raise ValueError("Learning Rate stategy not recognized")

  warmup_epochs = lr_params.get('warmup_epochs', None)
  if warmup_epochs:
    logging.info('Learning rate warmup_epochs: %d', warmup_epochs)
    warmup_steps = int(warmup_epochs * num_step_by_epoch)
    warmup_lr = (
      lr_params['learning_rate'] * tf.cast(global_step, tf.float32) / \
        tf.cast(warmup_steps, tf.float32))
    learning_rate = tf.cond(
      global_step < warmup_steps, lambda: warmup_lr, lambda: learning_rate)

  logging.info("Using '{}' strategy for learning rate".format(lr_strategy))
  tf_v1.summary.scalar('learning_rate', learning_rate)
  return learning_rate



