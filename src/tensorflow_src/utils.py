
import sys
import os
import shutil
import logging
import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2


def set_default_param_values_and_env_vars(params):
  """Sets up the default param values and environment variables ."""
  if params.batchnorm_persistent:
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  else:
    os.environ.pop('TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT', None)
  if params.winograd_nonfused:
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  else:
    os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
  os.environ['TF_SYNC_ON_FINISH'] = str(int(params.sync_on_finish))

  # Sets GPU thread settings
  if params.device == 'gpu':
    if params.gpu_thread_mode not in ['global', 'gpu_shared', 'gpu_private']:
      raise ValueError('Invalid gpu_thread_mode: %s' % params.gpu_thread_mode)
    os.environ['TF_GPU_THREAD_MODE'] = params.gpu_thread_mode

    if params.per_gpu_thread_count and params.gpu_thread_mode == 'global':
      raise ValueError(
          'Invalid per_gpu_thread_count with gpu_thread_mode=global: %s' %
          params.per_gpu_thread_count)
    # Default to two threads. One for the device compute and the other for
    # memory copies.
    per_gpu_thread_count = params.per_gpu_thread_count or 2
    total_gpu_thread_count = per_gpu_thread_count * params.num_gpus

    if params.gpu_thread_mode == 'gpu_private':
      os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
    elif params.gpu_thread_mode == 'gpu_shared':
      os.environ['TF_GPU_THREAD_COUNT'] = str(total_gpu_thread_count)

    cpu_count = multiprocessing.cpu_count()
    if not params.num_inter_threads and params.gpu_thread_mode in [
        'gpu_private', 'gpu_shared'
    ]:
      main_thread_count = max(cpu_count - total_gpu_thread_count, 1)
      params.num_inter_threads = main_thread_count

    # From the total cpu thread count, subtract the total_gpu_thread_count,
    # and then 2 threads per GPU device for event monitoring and sending /
    # receiving tensors
    num_monitoring_threads = 2 * params.num_gpus
    num_private_threads = max(
        cpu_count - total_gpu_thread_count - num_monitoring_threads, 1)
    if params.datasets_num_private_threads == 0:
      params.datasets_num_private_threads = num_private_threads
  return params


def create_config_proto(params):
  """Returns session config proto.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.log_device_placement = params.log_device_placement
  if params.num_intra_threads is None:
    if params.num_gpus:
      config.intra_op_parallelism_threads = 1
  else:
    config.intra_op_parallelism_threads = params.num_intra_threads
  if params.xla:
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  config.inter_op_parallelism_threads = params.num_inter_threads
  config.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
  # config.gpu_options.experimental.collective_ring_order = self.params.gpu_indices

  # TODO(b/117324590): Re-enable PinToHostOptimizer when b/117324590 is fixed.
  # Currently we have to disable PinToHostOptimizer w/ XLA since it causes
  # OOM/perf cliffs.
  config.graph_options.rewrite_options.pin_to_host_optimization = (
      rewriter_config_pb2.RewriterConfig.OFF)
  return config




def make_summary(name, value, summary_writer, global_step):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  summary_writer.add_summary(summary, global_step)



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


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias





