
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

import attacks
from . import layers

from config import hparams as FLAGS

class LogisticModel:

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):
    x = tf.layers.flatten(model_input)
    logits = tf.layers.dense(x, 10, activation=None, use_bias=True)
    return logits


class LogisticModelAttack:

  def create_model(self, x, n_classes, is_training, *args, **kwargs):
    """Build the core model within the graph."""

    self.is_training = is_training
    model = LogisticModel()

    if not is_training:
      with tf.variable_scope('network_defense'):
        return model.create_model(x, n_classes, is_training)

    else:
      labels = kwargs['labels']

      self.config = FLAGS.train_attack
      attack_method = self.config['attack_method']
      attack_cls = getattr(attacks, attack_method, None)
      if attack_cls is None:
        raise ValueError("Attack is not recognized.")
      attack_config = self.config[attack_method]
      self.attack = attack_cls(**attack_config)

      # get loss and logits from adv examples
      def fn_logits(x):
        with tf.variable_scope('network_defense', reuse=tf.AUTO_REUSE):
          return model.create_model(x, n_classes, is_training)

      logits = fn_logits(x)
      adv = self.attack.generate(x, fn_logits)

      logits_under_attack = fn_logits(adv)
      return logits, logits_under_attack
