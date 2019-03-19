
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

import attacks
from .base import BaseModel
from .wide_resnet import WideResnetModel

from config import hparams as FLAGS


class DefenseVsAttack(BaseModel):

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



class ImaginaryDefense(BaseModel):

  def create_model(self, x, n_classes, is_training, *args, **kwargs):
    """Build the core model within the graph."""

    self.is_training = is_training
    config = FLAGS.imaginary_defense
    use_orig = config['use_orig']
    use_real = config['use_real']
    use_imag = config['use_imag']
    share_weights = config['share_weights']


    x_complex = tf.cast(x, tf.complex64)
    x_fft = tf.spectral.fft(x_complex)
    x_real = tf.real(x_fft)
    x_imag = tf.imag(x_fft)
    if share_weights:
      logits_array = []
      resnet_defense = WideResnetModel()
      with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
        if use_orig:
          logits = resnet_defense.create_model(x, n_classes, is_training)
          logits_array.append(logits)
        if use_real:
          logits_real = resnet_defense.create_model(x_real, n_classes, is_training)
          logits_array.append(logits_real)
        if use_imag:
          logits_imag = resnet_defense.create_model(x_imag, n_classes, is_training)
          logits_array.append(logits_imag)

    elif not share_weights:
      logits_array = []
      resnet_defense = WideResnetModel()
      if use_orig:
        with tf.variable_scope('network'):
          logits = resnet_defense.create_model(x, n_classes, is_training)
          logits_array.append(logits)
      if use_real:
        with tf.variable_scope('network_real'):
          logits_real = resnet_defense.create_model(x_real, n_classes, is_training)
          logits_array.append(logits_real)
      if use_imag:
        with tf.variable_scope('network_imag'):
          logits_imag = resnet_defense.create_model(x_imag, n_classes, is_training)
          logits_array.append(logits_imag)

    final_logits = tf.reduce_mean(logits_array, 0)
    return final_logits




