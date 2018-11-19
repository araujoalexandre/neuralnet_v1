"""Provides definitions for non-regularized training or test losses."""

import tensorflow as tf


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


class SoftmaxCrossEntropyWithLogits:

  def calculate_loss(self, predictions, labels, **unused_params):
    return tf.losses.sparse_softmax_cross_entropy(
          labels=labels, logits=predictions)
