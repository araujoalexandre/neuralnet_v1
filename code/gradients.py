"""Contains a collection of util functions for training and evaluating.
"""
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from config import hparams as FLAGS


class ProcessGradients:

  def __init__(self, tower_grads):
    self.tower_grads = tower_grads

  def _clip_gradient_norms(self, gradients_to_variables, max_norm):
    """Clips the gradients by the given value.

    Args:
      gradients_to_variables: A list of gradient to variable pairs (tuples).
      max_norm: the maximum norm value.

    Returns:
      A list of clipped gradient to variable pairs.
    """
    clipped_grads_and_vars = []
    for grad, var in gradients_to_variables:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          tmp = tf.clip_by_norm(grad.values, max_norm)
          grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
        else:
          grad = tf.clip_by_norm(grad, max_norm)
      clipped_grads_and_vars.append((grad, var))
    return clipped_grads_and_vars

  def _combine_gradients(self):
    """Calculate the combined gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been summed
       across all towers.
    """
    filtered_grads = [[x for x in grad_list if x[0] is not None]
    for grad_list in self.tower_grads]
    final_grads = []
    for i in range(len(filtered_grads[0])):
      grads = [filtered_grads[t][i] for t in range(len(filtered_grads))]
      grad = tf.stack([x[0] for x in grads], 0)
      grad = tf.reduce_sum(grad, 0)
      final_grads.append((grad, filtered_grads[0][i][1],))
    return final_grads

  def get_gradients(self):
    gradients = self._combine_gradients()
    if FLAGS.clip_gradient_norm > 0:
      with tf.name_scope('clip_grads'):
        gradients = self._clip_gradient_norms(
          gradients, FLAGS.clip_gradient_norm)
    return gradients

