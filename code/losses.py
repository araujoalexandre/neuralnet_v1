"""Provides definitions for non-regularized training or test losses."""

import tensorflow as tf

from config import hparams as FLAGS

class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

     Args:
      unused_predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      unused_labels: a 2-d tensor storing the labels, which has the same shape
        as the unused_predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()


class SoftmaxCrossEntropyWithLogits(BaseLoss):

  def calculate_loss(self, labels=None, logits=None, **unused_params):
    with tf.name_scope("loss"):
      if FLAGS.gradients["compute_hessian"]:
        if not FLAGS.one_hot_labels:
          raise ValueError("Labels needs to be one hot encoded.")
        # We are going to compute the hessian so we can't 
        # use the tf.losses.sparse_softmax_cross_entropy
        # because the ops are fused and the gradient of the 
        # cross entropy is blocked. Insted we compute it manually.
        # /!\ might not be efficient and not numerically stable
        labels = tf.cast(labels, tf.float32)
        proba = tf.nn.softmax(logits)
        loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(proba), axis=1))
        return loss
      return tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

