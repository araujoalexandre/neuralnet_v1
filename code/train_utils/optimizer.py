
import tensorflow as tf
from tensorflow import logging

from config import hparams as FLAGS


class Optimizer:

  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

  def GradientDescentOptimizer(self):
    opt = tf.train.GradientDescentOptimizer(self.learning_rate,
                                            use_locking=False)
    return opt

  def MomentumOptimizer(self):
    config = FLAGS.MomentumOptimizer
    opt = tf.train.MomentumOptimizer(self.learning_rate, config['momentum'],
                                     use_locking=config['use_locking'])
    return opt

  def RMSPropOptimizer(self):
    config = FLAGS.RMSPropOptimizer
    RMSProp_decay = config['RMSProp_decay']
    RMSProp_momentum = config['RMSProp_momentum']
    RMSProp_epsilon = config['RMSProp_epsilon']
    opt = tf.train.RMSPropOptimizer(self.learning_rate, RMSProp_decay,
                                    momentum=RMSProp_momentum,
                                    epsilon=RMSProp_epsilon)
    return opt

  def AdamOptimizer(self):
    config = FLAGS.AdamOptimizer
    opt = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate,
      beta1=config['beta1'],
      beta2=config['beta2'],
      epsilon=config['epsilon'],
      use_locking=config['use_locking'])
    return opt

  def get_optimizer(self):
    logging.info("Using '{}' as optimizer".format(FLAGS.optimizer))
    opt = getattr(self, FLAGS.optimizer)()
    return opt

