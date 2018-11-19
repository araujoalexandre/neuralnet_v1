
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging
from tensorflow import convert_to_tensor as to_tensor
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer

FLAGS = flags.FLAGS

# global 
flags.DEFINE_string("optimizer", "AdamOptimizer",
                    "What optimizer class to use.")
flags.DEFINE_bool("use_locking", False, 
                  "If True use locks for update operations.")

# MomentumOptimizer
flags.DEFINE_float("momentum", 0.9, 
                   "A Tensor or a floating point value. The momentum.")
flags.DEFINE_bool("use_nesterov", False, 
                  "If True use Nesterov Momentum.")

# AdamOptimizer
flags.DEFINE_float("beta1", 0.9, 
                   "A float value or a constant float tensor. The exponential "
                   "decay rate for the 1st moment estimates.")
flags.DEFINE_float("beta2", 0.999, 
                   "A float value or a constant float tensor. The exponential "
                   "decay rate for the 2nd moment estimates.")
flags.DEFINE_float("epsilon", 1e-08, 
                   "A small constant for numerical stability.")


class Optimizer:

  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

  def MomentumOptimizer(self):
    opt = tf.train.MomentumOptimizer(self.learning_rate, FLAGS.momentum)
    return opt

  def AdamOptimizer(self):
    opt = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate, 
      beta1=FLAGS.beta1,
      beta2=FLAGS.beta2,
      epsilon=FLAGS.epsilon,
      use_locking=FLAGS.use_locking)
    return opt

  def get_optimizer(self):
    logging.info("Using '{}' as optimizer".format(FLAGS.optimizer))
    # opt = getattr(self, FLAGS.optimizer)()

    opt = tf.train.MomentumOptimizer(self.learning_rate, FLAGS.momentum)
    return opt
