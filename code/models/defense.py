
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

import attacks
from .wide_resnet import WideResnetModel

from config import hparams as FLAGS


class DefenseVsAttack:

  def create_model(self, x, n_classes, is_training, *args, **kwargs):
    """Build the core model within the graph."""

    self.is_training = is_training
    resnet_defense = WideResnetModel()

    if not is_training:
      with tf.variable_scope('network_defense'):
        return resnet_defense.create_model(x, n_classes, is_training)

    else:
      labels = kwargs['labels']

      reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
      if reg_fn is None:
        self.regularizer = None
      else:
        self.regularizer = reg_fn(l=FLAGS.weight_decay_rate)

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
          return resnet_defense.create_model(x, n_classes, is_training)

      logits = fn_logits(x)
      adv = self.attack.generate(x, fn_logits)

      if self.config['noise_attack']:
        # generate noise
        logging.info('noise attack activated')
        shape = adv.shape.as_list()[1:]
        loc = tf.zeros(tf.shape(adv), dtype=tf.float32)
        scale = tf.ones(tf.shape(adv), dtype=tf.float32)
        noise = tf.distributions.Normal(loc, scale).sample()
        if self.config['learn_noise_attack']:
          # learn noise
          logging.info('parameterized noise attack activated')
          with tf.variable_scope('network_attack', reuse=tf.AUTO_REUSE):
            noise = tf.layers.flatten(noise)
            feature_size = noise.shape.as_list()[-1]
            weights = tf.get_variable('weights_noise', (feature_size, ),
              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/feature_size)),
              regularizer=self.regularizer)
            biases = tf.get_variable('biases_noise', (feature_size, ),
              regularizer=self.regularizer)
            noise = tf.multiply(noise, weights) + biases
            noise = tf.reshape(noise, (-1, *shape))
            perturbation = x - adv
            noise = attack_config['eps'] * tf.nn.tanh(perturbation + noise)
        # inject noise in adv
        adv = adv + noise

      logits_under_attack = fn_logits(adv)
      return logits, logits_under_attack



