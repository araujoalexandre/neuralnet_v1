"""Contains a collection of util functions for training and evaluating.
"""
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from config import hparams as FLAGS


class ComputeAndProcessGradients:

  def __init__(self):
    self.config = FLAGS.gradients

  def _clip_gradient_norms(self, gradients_to_variables, max_norm):
    """Clips the gradients by the given value.

    Args:
      gradients_to_variables: A list of gradient to variable pairs (tuples).
      max_norm: the maximum norm value.

    Returns:
      A list of clipped gradient to variable pairs.
    """
    with tf.name_score('clip_gradients'):
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

  def _perturbed_gradients(self, gradients):
    with tf.name_scope('perturbed_gradients'):
      gradients_norm = tf.constant(0.)
      for grad, var in gradients:
        if grad is not None:
          norm = tf.norm(grad)
          gradients_norm = gradients_norm + norm
          tf.summary.scalar(var.op.name + '/gradients/norm', norm)
      gradients_norm = gradients_norm / len(gradients)
      tf.add_to_collection("gradients_norm", gradients_norm)

      if not self.config['perturbed_gradients']:
        return gradients

      norm_threshold = self.config['perturbed_threshold']
      activate_noise = tf.cond(tf.less(gradients_norm, norm_threshold),
                       true_fn=lambda: tf.constant(1.),
                       false_fn=lambda: tf.constant(0.))

      gradients_noise = []
      for grad, var in gradients:
        Y = tf.random_normal(shape=grad.shape)
        U = tf.random_uniform(shape=grad.shape, minval=0, maxval=1)
        noise = tf.sqrt(U) * (Y / tf.norm(Y)) * activate_noise
        tf.summary.histogram(var.op.name + '/gradients/noise', noise)
        grad = grad + noise
        gradients_noise.append((grad, var))
      return gradients_noise

  def _gradients_summary(self, gradients):
    for grad, var in gradients:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

  def _compute_hessian_and_summary(self, losses):
    """Compute the hessian of the training variables"""
    with tf.name_scope("hessian"):
      var_list = tf.trainable_variables()
      tower_hessians = []
      for loss in losses:
        hessians = tf.hessians(loss, var_list)
        tower_hessians.append(hessians)
      hessians = self._combine_gradients(tower_hessians)
      for hess, var in zip(hessians, var_list):
        eigenvalues = tf.linalg.svd(hess, compute_uv=False, full_matrices=False)
        tf.summary.histogram(var.op.name + '/hessian', hess)
        tf.summary.histogram(var.op.name + '/hessian/eigenvalues', eigenvalues)

  def get_gradients(self, opt, losses):

    gradients = opt.compute_gradients(loss)
    self._gradients_summary(gradients)

    # compute and record summary of hessians eigenvals
    if self.config['compute_hessian']:
      self._compute_hessian_and_summary(losses)

    # to help convergence, inject noise in gradients
    gradients = self._perturbed_gradients(gradients)

    # to regularize, clip the value of the gradients
    if self.config['clip_gradient_norm'] > 0:
      with tf.name_scope('clip_grads'):
        gradients = self._clip_gradient_norms(
          gradients, self.config['clip_gradient_norm'])
    return gradients

