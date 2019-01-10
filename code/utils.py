
import numpy as np
import tensorflow as tf


def make_summary(name, value, summary_writer, global_step_val):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  summary_writer.add_summary(summary, global_step_val)

def summary_histogram(writer, tag, values, step, bins=1000):

  # Create histogram using numpy
  counts, bin_edges = np.histogram(values, bins=bins)

  # Fill fields of histogram proto
  hist = tf.HistogramProto()
  hist.min = float(np.min(values))
  hist.max = float(np.max(values))
  hist.num = int(np.prod(values.shape))
  hist.sum = float(np.sum(values))
  hist.sum_squares = float(np.sum(values**2))

  # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
  # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
  # Thus, we drop the start of the first bin
  bin_edges = bin_edges[1:]

  # Add bin edges and counts
  for edge in bin_edges:
      hist.bucket_limit.append(edge)
  for c in counts:
      hist.bucket.append(c)

  # Create and write Summary
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
  writer.add_summary(summary, step)
  writer.flush()


def l1_normalize(x, dim, epsilon=1e-12, name=None):
  """Normalizes along dimension `dim` using an L1 norm.
  For a 1-D tensor with `dim = 0`, computes
      output = x / max(sum(abs(x)), epsilon)
  For `x` with more dimensions, independently normalizes each 1-D slice along
  dimension `dim`.
  Args:
    x: A `Tensor`.
    dim: Dimension along which to normalize.  A scalar or a vector of
      integers.
    epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
      divisor if `norm < sqrt(epsilon)`.
    name: A name for this operation (optional).
  Returns:
    A `Tensor` with the same shape as `x`.
  """
  with tf.name_scope(name, "l1_normalize", [x]) as name:
    abs_sum = tf.reduce_sum(tf.abs(x), dim, keep_dims = True)
    x_inv_norm = tf.reciprocal(tf.maximum(abs_sum, epsilon))
    return tf.multiply(x, x_inv_norm, name=name)
