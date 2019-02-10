
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from config import hparams as FLAGS

class UpdateOps:

  def __init__(self, optimizer, gradients, global_step):
    self.config = FLAGS.update_ops
    self.optimizer = optimizer
    self.gradients = gradients
    self.global_step = global_step

  def _parseval_update(self):
    beta = self.config['parseval_step']
    parseval_convs = tf.get_collection("parseval_convs")
    parseval_dense = tf.get_collection("parseval_dense")

    for kernel in parseval_convs:
      shape = kernel.get_shape().as_list()
      w_t = tf.reshape(kernel, [-1, shape[-1]])
      w = tf.transpose(w_t)
      for _ in range(self.config['parseval_loops']):
        w   = (1 + beta) * w - beta * tf.matmul(w, tf.matmul(w_t, w))
        w_t = tf.transpose(w)
      op = tf.assign(kernel, tf.reshape(w_t, shape), validate_shape=True)
      tf.add_to_collection("ops_after_update", op)

    for _W in parseval_dense:
      w_t = _W
      w = tf.transpose(w_t)
      for _ in range(self.config['parseval_loops']):
        w = (1 + beta) * w - beta * tf.matmul(w, tf.matmul(w_t, w))
        w_t = tf.transpose(w)
      op = tf.assign(_W, w_t, validate_shape=True)
      tf.add_to_collection("ops_after_update", ops)


  def make_update(self):

    ops_before_update = tf.get_collection("ops_before_update")
    ops_before_update = tf.group(*ops_before_update)
    with tf.control_dependencies([ops_before_update]):
      train_op = self.optimizer.apply_gradients(self.gradients,
        global_step=self.global_step)

    if self.config['parseval_update']:
      self._parseval_update()

    ops_after_update = tf.get_collection("ops_after_update")
    ops_after_update = tf.group(*ops_after_update)
    with tf.control_dependencies([ops_after_update]):
        train_op = tf.no_op()

    return train_op
