
import multiprocessing


def set_default_param_values_and_env_vars(params):
  """Sets up the default param values and environment variables ."""
  # Sets GPU thread settings
  if params.device == 'gpu':
    if params.gpu_thread_mode not in ['global', 'gpu_shared', 'gpu_private']:
      raise ValueError('Invalid gpu_thread_mode: %s' % params.gpu_thread_mode)

    if params.per_gpu_thread_count and params.gpu_thread_mode == 'global':
      raise ValueError(
          'Invalid per_gpu_thread_count with gpu_thread_mode=global: %s' %
          params.per_gpu_thread_count)
    # Default to two threads. One for the device compute and the other for
    # memory copies.
    per_gpu_thread_count = params.per_gpu_thread_count or 2
    total_gpu_thread_count = per_gpu_thread_count * params.num_gpus

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
