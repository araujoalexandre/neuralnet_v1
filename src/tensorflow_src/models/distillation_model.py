
import logging

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

from . import model as model_lib
from . import resnet_model
from . import circulant_model


class DistillationModel(model_lib.CNNModel):

  def __init__(self, teacher, student, params):
    super(DistillationModel, self).__init__(
        'DistillationModel', params=params)
    self.teacher = teacher(params)
    self.student = student(params)

  def skip_final_affine_layer(self):
    return True

  def add_inference(self, cnn):

    nclass = cnn.nclass
    self.is_training = cnn.phase_train
    input_ = [cnn.top_layer, None]
    if self.is_training:
      with tf.name_scope('teacher'):
        build_network_result = self.teacher.build_network(
          input_, self.is_training, nclass=nclass)
        self.logits_t = build_network_result.logits

    with tf.name_scope('student'):
      build_network_result = self.student.build_network(
      input_, self.is_training, nclass=nclass)
      self.logits_s = build_network_result.logits
      cnn.top_layer = self.logits_s

  def _loss_function_eval(self, inputs, build_network_result):
    """Returns the op to measure the loss of the model."""
    logits = build_network_result.logits
    _, labels = inputs
    with tf.name_scope('xentropy'):
      cross_entropy = tf_v1.losses.sparse_softmax_cross_entropy(
          logits=logits, labels=labels)
      loss = tf.reduce_mean(cross_entropy, name='xentropy1_mean')
    return loss

  def _loss_function_train(self, inputs, build_network_result):
    """Returns the op to measure the loss of the model."""
    logits_t, logits_s = self.logits_t, self.logits_s
    _, labels = inputs
    with tf.name_scope('xentropy'):
      cross_entropy = tf_v1.losses.sparse_softmax_cross_entropy(
          logits=logits_t, labels=labels)
      loss1 = tf.reduce_mean(cross_entropy, name='xentropy1_mean')

      cross_entropy = tf_v1.losses.sparse_softmax_cross_entropy(
          logits=logits_s, labels=labels)
      loss2 = tf.reduce_mean(cross_entropy, name='xentropy2_mean')

      cross_entropy = tf_v1.losses.softmax_cross_entropy(
          logits=logits_s, onehot_labels=tf.nn.softmax(logits_t))
      loss3 = tf.reduce_mean(cross_entropy, name='xentropy3_mean')

    total_loss = loss1 + loss2 + loss3
    return total_loss

  def loss_function(self, *args):
    """Returns the op to measure the loss of the model."""
    if self.is_training:
      return self._loss_function_train(*args)
    else:
      return self._loss_function_eval(*args)


def create_distillation_resnet56_v2_circulant(params):
  teacher = resnet_model.create_resnet56_v2_cifar_model
  student = circulant_model.DiagonalCirculantModel
  return DistillationModel(teacher, student, params)


