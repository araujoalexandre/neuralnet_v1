
import os, sys
import json
import time
import pprint
import socket
# import logging as pylogging
import multiprocessing
from collections import OrderedDict
from collections import namedtuple
from os.path import join, exists

import models
from models import model
from models import model_config
from dataset import readers
from train_utils import losses
from train_utils.learning_rate import LearningRate
from train_utils.optimizer import Optimizer
from train_utils.gradients import ComputeAndProcessGradients
from train_utils.gradients import compute_hessian_and_summary
from train_utils.gradients import combine_gradients
from train_utils.update_ops import UpdateOps
from train_utils import variable_mgr, variable_mgr_util
from eval_utils import eval_util
from utils import MessageBuilder

import numpy as np
import tensorflow as tf
from tensorflow import app
from tensorflow import gfile
from tensorflow import logging
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.util import nest

from config import hparams as FLAGS

# InputProcessingInfo contains various sources of inputs which will be later fed
# into the model. If synthetic data is used, all three fields are None.
InputProcessingInfo = namedtuple(
    'InputProcessingInfo',
    [
        # The first two fields are non-None iff datasets prefetching is not
        # used.

        # Ops that produce the input batches.
        'input_producer_op',
        # A list of StagingArea for each device.
        'input_producer_stages',

        # Input produced using multi device iterator. Non-None iff datasets
        # prefetching is used
        'multi_device_iterator_input'
    ])

def task_as_string(task):
  return "/job:{}/task:{}".format(task.type, task.index)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def maybe_compile(computation, params):
  if params and params.xla_compile:
    return tf.xla.experimental.compile(computation)
  else:
    return computation()


def set_default_param_values_and_env_vars(params):
  """Sets up the default param values and environment variables ."""
  if True: # params.batchnorm_persistent:
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  else:
    os.environ.pop('TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT', None)
  if True: # params.winograd_nonfused:
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  else:
    os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
  if None: # params.autotune_threshold:
    os.environ['TF_AUTOTUNE_THRESHOLD'] = str(params.autotune_threshold)
  os.environ['TF_SYNC_ON_FINISH'] = '0' # str(int(params.sync_on_finish))
  # argparse.ArgumentParser(
  #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Sets GPU thread settings
  if FLAGS.device == 'gpu':
    # params = FLAGS._replace(gpu_thread_mode=FLAGS.gpu_thread_mode)
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
    total_gpu_thread_count = per_gpu_thread_count * params.train_num_gpus

    if params.gpu_thread_mode == 'gpu_private':
      os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
    elif params.gpu_thread_mode == 'gpu_shared':
      os.environ['TF_GPU_THREAD_COUNT'] = str(total_gpu_thread_count)

    cpu_count = multiprocessing.cpu_count()
    if not params.num_inter_threads and params.gpu_thread_mode in [
        'gpu_private', 'gpu_shared'
    ]:
      main_thread_count = max(cpu_count - total_gpu_thread_count, 1)
      # params = params._replace(num_inter_threads=main_thread_count)
      params.num_inter_threads = main_thread_count

    # From the total cpu thread count, subtract the total_gpu_thread_count,
    # and then 2 threads per GPU device for event monitoring and sending /
    # receiving tensors
    num_monitoring_threads = 2 * params.train_num_gpus
    num_private_threads = max(
        cpu_count - total_gpu_thread_count - num_monitoring_threads, 1)
    # params = params._replace(datasets_num_private_threads=num_private_threads)
    params.datasets_num_private_threads = num_private_threads
  return params


def setup():
  """Sets up the environment that BenchmarkCNN should run in.

  Returns:
    A potentially modified params.
  Raises:
    ValueError: invalid parames combinations.
  """
  # Set up environment variables before doing any other global initialization to
  # make sure it uses the appropriate environment variables.
  set_default_param_values_and_env_vars(FLAGS)
  # platforms_util.initialize(params, create_config_proto(params))
  if FLAGS.job_name:
    # Create a dummy session to initialize TF global variables using the input
    # params. Otherwise, ListDevices function may create global devices using
    # the default config instead of using the user provided config.
    #
    # TODO(hinsu): Find a way to achieve the same for distributed benchmark. It
    # is not legal to create distributed session after local session. It is also
    # not possible to create distributed session here as that results in
    # multiple creation of ClusterManager and Server.
    with tf.Session(config=create_config_proto(FLAGS)) as sess:
      del sess


def create_config_proto():
  """Returns session config proto.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  if FLAGS.num_intra_threads is None:
    if FLAGS.train_num_gpus:
      config.intra_op_parallelism_threads = 1
  else:
    config.intra_op_parallelism_threads = FLAGS.num_intra_threads
  if FLAGS.xla:
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  config.inter_op_parallelism_threads = FLAGS.num_inter_threads
  config.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
  # config.gpu_options.experimental.collective_ring_order = FLAGS.gpu_indices

  # TODO(b/117324590): Re-enable PinToHostOptimizer when b/117324590 is fixed.
  # Currently we have to disable PinToHostOptimizer w/ XLA since it causes
  # OOM/perf cliffs.
  config.graph_options.rewrite_options.pin_to_host_optimization = (
      rewriter_config_pb2.RewriterConfig.OFF)
  return config


def _get_checkpoint_to_load(ckpt_dir):
  """Returns which checkpoint to load.

  Args:
    ckpt_dir: Path to a folder of checkpoints or full path to a checkpoint.

  Returns:
    Full path to checkpoint to load.

  Raises:
    CheckpointNotFoundException: If checkpoint is not found.
  """
  p = re.compile(r'ckpt-\d+$')
  if p.search(ckpt_dir):
    model_checkpoint_path = ckpt_dir
  else:
    # Finds latest checkpoint in directory provided
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      model_checkpoint_path = ckpt.model_checkpoint_path
    else:
      raise CheckpointNotFoundException('No checkpoint file found in dir:{}'.
                                        format(ckpt_dir))
  return model_checkpoint_path



class Trainer:
  """A Trainer to train a Tensorflow graph."""

  def __init__(self):
    """Creates a Trainer.
    """
    self.job_name = FLAGS.job_name
    self.task_index = FLAGS.task_index
    self.is_master = (self.job_name in ('', 'worker') and self.task_index == 0)
    self.start_new_model = FLAGS.start_new_model
    self.train_dir = FLAGS.train_dir
    self.num_gpus = FLAGS.train_num_gpus
    self.variable_update = FLAGS.variable_update

    self.batch_size = FLAGS.train_batch_size * self.num_gpus
    if self.job_name:
      self.global_batch_size = self.batch_size * \
          (self.cluster.num_tasks('worker') + 1)
    else:
      self.global_batch_size = self.batch_size

    # logging.info("Using total batch size of {} for training "
    #   "over {} GPUs: batch size of {} per GPUs.".format(
    #     batch_size, num_towers, batch_size // num_towers))


    # PS server is used for distributed jobs not using all-reduce.
    use_ps_server = self.job_name and \
      (self.variable_update != 'distributed_all_reduce' and \
       self.variable_update != 'collective_all_reduce')

    # controller is used for distributed_all_reduce with > 1 worker.
    use_controller = (
        self.variable_update == 'distributed_all_reduce' and
        self.job_name)
    if use_controller and not FLAGS.controller_host:
      raise ValueError('When variable_update==distributed_all_reduce '
                       'controller_host must also be specified.')
    # collective_all_reduce doesn't need a controller or ps
    self.distributed_collective = (
        self.variable_update == 'collective_all_reduce' and
        self.job_name)

    self.local_parameter_device_flag = FLAGS.local_parameter_device
    if self.job_name:
      self.cluster_manager = platforms_util.get_cluster_manager(
          params, create_config_proto(params))
      assert isinstance(self.cluster_manager, cnn_util.BaseClusterManager)

      worker_prefix = '/job:worker/replica:0/task:{}'.format(self.task_index)
      if use_ps_server:
        self.param_server_device = tf.train.replica_device_setter(
            worker_device=worker_prefix + '/cpu:0',
            cluster=self.cluster_manager.get_cluster_spec())
        # This device on which the queues for managing synchronization between
        # servers should be stored.
        self.sync_queue_devices = [
            '/job:ps/replica:0/task:{}/cpu:0'.format(i)
            for i in range(self.cluster_manager.num_ps())
        ]
      else:
        self.sync_queue_devices = ['/job:worker/replica:0/task:0/cpu:0']
    else:
      self.task_index = 0
      self.cluster_manager = None
      worker_prefix = ''
      self.param_server_device = '/{}:0'.format(FLAGS.local_parameter_device)
      self.sync_queue_devices = [self.param_server_device]

    if self.cluster_manager:
      self.num_workers = self.cluster_manager.num_workers()
    elif FLAGS.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      self.num_workers = hvd.size()
    else:
      self.num_workers = 1
    self.num_ps = self.cluster_manager.num_ps() if self.cluster_manager else 0

    if self.num_workers > 1 and FLAGS.all_reduce_spec == 'nccl':
      raise ValueError('--all_reduce_spec=nccl is invalid in a '
                       'multi-worker job')

    # Device to use for ops that need to always run on the local worker's CPU.
    self.cpu_device = '{}/cpu:0'.format(worker_prefix)

    # Device to use for ops that need to always run on the local worker's
    # compute device, and never on a parameter server device.
    self.raw_devices = [
        '{}/gpu:{}'.format(worker_prefix, i)
        for i in range(self.num_gpus)
    ]

    if FLAGS.variable_update == 'parameter_server':
      if self.job_name:
        if not FLAGS.staged_vars:
          self.variable_mgr = variable_mgr.VariableMgrDistributedFetchFromPS(
              self)
        else:
          self.variable_mgr = (
              variable_mgr.VariableMgrDistributedFetchFromStagedPS(self))
      else:
        if not FLAGS.staged_vars:
          self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
        else:
          self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromStagedPS(
              self)
    elif self.variable_update == 'replicated':
      if self.job_name:
        raise ValueError('Invalid variable_update in distributed mode: %s' %
                         self.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
          self, FLAGS.all_reduce_spec,
          FLAGS.agg_small_grads_max_bytes,
          FLAGS.agg_small_grads_max_group,
          FLAGS.allreduce_merge_scope)
    elif self.variable_update == 'distributed_all_reduce':
      assert FLAGS.cross_replica_sync
      self.variable_mgr = variable_mgr.VariableMgrDistributedAllReduce(
          self, FLAGS.all_reduce_spec,
          ('worker' if self.num_workers > 1 else 'localhost'),
          self.num_workers, FLAGS.agg_small_grads_max_bytes,
          FLAGS.agg_small_grads_max_group,
          FLAGS.allreduce_merge_scope)
    elif FLAGS.variable_update == 'collective_all_reduce':
      assert FLAGS.cross_replica_sync
      self.variable_mgr = variable_mgr.VariableMgrCollectiveAllReduce(
          self, FLAGS.all_reduce_spec,
          self.num_workers, self.num_gpus, self.task_index,
          FLAGS.allreduce_merge_scope)
    elif self.variable_update == 'distributed_replicated':
      assert FLAGS.cross_replica_sync
      if not self.job_name:
        raise ValueError('Invalid variable_update in local mode: {}'.format(
                         self.variable_update))
      self.variable_mgr = variable_mgr.VariableMgrDistributedReplicated(self)
    elif FLAGS.variable_update in ('independent', 'horovod'):
      if self.job_name:
        raise ValueError(
          'Invalid variable_update in distributed mode: {}'.format(
                         self.variable_update))
      self.variable_mgr = variable_mgr.VariableMgrIndependent(self)
    else:
      raise ValueError(
          'Invalid variable_update: {}'.format(self.variable_update))

    # Device to use for running on the local worker's compute device, but
    # with variables assigned to parameter server devices.
    self.devices = self.variable_mgr.get_devices()
    if self.job_name:
      if use_ps_server:
        self.global_step_device = self.param_server_device
      elif FLAGS.variable_update == 'collective_all_reduce':
        self.global_step_device = self.cpu_device
      else:
        self.global_step_device = '/job:worker/replica:0/task:0/cpu:0'
    else:
      self.global_step_device = self.cpu_device


    # TODO: remove auto loss scale and check inf in grad
    self.enable_auto_loss_scale = False

    # create Session Proto configuration
    self.sess_config = create_config_proto()

    self.model = model_config.get_model_config(
        FLAGS.model, FLAGS.dataset, FLAGS)
    # self.model = find_class_by_name(FLAGS.model, [models])()
    self.reader = find_class_by_name(FLAGS.reader, [readers])(
      self.batch_size, self.raw_devices, self.cpu_device, is_training=True)

    # define the number of steps
    num_steps_by_epochs = self.reader.n_train_files / self.global_batch_size
    self.max_steps = FLAGS.num_epochs * num_steps_by_epochs


  def add_forward_pass_and_gradients(self,
                                     rel_device_num,
                                     abs_device_num,
                                     input_processing_info,
                                     gpu_compute_stage_ops,
                                     gpu_grad_stage_ops):
    """Add ops for forward-pass and gradient computations."""
    n_classes = self.reader.n_classes
    assert input_processing_info.multi_device_iterator_input, (
        'multi_device_iterator_input cannot be None if '
        'datasets_use_prefetch=True')
    input_list = (
        input_processing_info.multi_device_iterator_input[rel_device_num])

    def forward_pass_and_gradients():
      """Builds forward pass and gradient computation network.

      Returns:
        outputs: A list of tensors depending on different modes.
      """
      build_network_result = self.model.build_network(
        input_list, True, n_classes)
      logits = build_network_result.logits

      base_loss = self.model.loss_function(input_list, build_network_result)
      params = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num)
      l2_loss = None
      total_loss = base_loss
      with tf.name_scope('l2_loss'):
        filtered_params = self.model.filter_l2_loss_vars(params)
        if rel_device_num == len(self.devices) - 1:
          # We compute the L2 loss for only one device instead of all of them,
          # because the L2 loss for each device is the same. To adjust for this,
          # we multiply the L2 loss by the number of devices. We choose the
          # last device because for some reason, on a Volta DGX1, the first four
          # GPUs take slightly longer to complete a step than the last four.
          # TODO(reedwm): Shard the L2 loss computations across GPUs.
          if FLAGS.single_l2_loss_op:
            # TODO(reedwm): If faster, create a fused op that does the L2 loss
            # on multiple tensors, and use that instead of concatenating
            # tensors.
            reshaped_params = [tf.reshape(p, (-1,)) for p in filtered_params]
            l2_loss = tf.nn.l2_loss(tf.concat(reshaped_params, axis=0))
          else:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in filtered_params])
      weight_decay = FLAGS.weight_decay
      if (weight_decay is not None and weight_decay != 0. and
          l2_loss is not None):
        total_loss += len(self.devices) * weight_decay * l2_loss

      results = {}
      results['logits'] = logits
      results['loss'] = total_loss
      results['grads'] = tf.gradients(total_loss, params)
      param_refs = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num, writable=True)
      results['gradvars'] = list(zip(results['grads'], param_refs))
      return results

    with tf.device(self.devices[rel_device_num]):
      return maybe_compile(forward_pass_and_gradients, FLAGS)


  def _build_model(self):
    """Build the TensorFlow graph."""
    # Adjust seed so different workers start read different input files.
    if self.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      seed_adjustment = hvd.rank()
    else:
      seed_adjustment = 0
    tf.set_random_seed(FLAGS.tf_random_seed + seed_adjustment)
    np.random.seed(4321 + seed_adjustment)
    is_training = True

    losses, logits, grads = [], [], []
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.train.get_or_create_global_step()

    # Build the processing and model for the worker.
    input_producer_op = None
    with tf.name_scope("input"):
      input_processing_info = InputProcessingInfo(
          input_producer_op=None,
          input_producer_stages=None,
          multi_device_iterator_input=None)
      # TODO: add shift_ratio = 0
      input_processing_info = input_processing_info._replace(
        multi_device_iterator_input=self.reader.input_fn().get_next())

    update_ops = None
    staging_delta_ops = []

    for device_num in range(len(self.devices)):
      with tf.name_scope('tower_{}'.format(device_num)) as name_scope, (
          self.variable_mgr.create_outer_variable_scope(device_num)):
        results = self.add_forward_pass_and_gradients(
            device_num, device_num, input_processing_info,
            gpu_compute_stage_ops, gpu_grad_stage_ops)

        losses.append(results['loss'])
        logits.append(results['logits'])
        grads.append(results['gradvars'])

        if device_num == 0:
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
          # TODO: remove attributes
          assert not self.variable_mgr.staging_delta_ops

    enqueue_ops = []
    fetches = self._build_fetches(global_step, logits, losses, grads,
                                  enqueue_ops, update_ops)
    return (input_producer_op, enqueue_ops, fetches)



  def _build_model_single_session(self):
    """Build the TensorFlow graph for multiple replicas in a single_session.

    Returns:
      input_producer_op:
      enqueue_ops:
      fetches:

    Raises:
       ValueError: optimizer not recognized.

    Single session runs multiple model replicas as part of one large
    distributed graph, whose global execution is always step-synchronized.
    """
    # verify assumptions
    assert self.task_index == 0

    tf.set_random_seed(FLAGS.tf_random_seed)
    np.random.seed(4321)
    is_training = True

    losses, logits, grads = [], [], []
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.train.get_or_create_global_step()

    update_ops = []
    global_input_producer_op = []

    is_local = not self.job_name
    if is_local:
      assert self.num_workers == 1
    for task_num in range(self.num_workers):
      # Reset the devices that self.variable_mgr knows about to those
      # belonging to the next worker (task).
      self.reset_devices_for_task(task_num, is_local)

      # TODO: remove InputProcessingInfo
      with tf.name_scope("input"):
        input_processing_info = InputProcessingInfo(
            input_producer_op=None,
            input_producer_stages=None,
            multi_device_iterator_input=None)
        # TODO: add shift_ratio=(task_num / self.num_workers)
        # to reader option
        input_processing_info = input_processing_info._replace(
          multi_device_iterator_input=self.reader.input_fn().get_next())

      # Build the per-worker model replica.
      for rel_device_num in range(len(self.devices)):
        abs_device_num = task_num * len(self.devices) + rel_device_num
        with self.variable_mgr.create_outer_variable_scope(
            abs_device_num), tf.name_scope(
                'task_{}_tower_{}'.format(task_num, rel_device_num)) as name_scope:
          results = self.add_forward_pass_and_gradients(
              rel_device_num, abs_device_num,
              input_processing_info, gpu_compute_stage_ops, gpu_grad_stage_ops)

          losses.append(results['loss'])
          logits.append(results['logits'])
          grads.append(results['gradvars'])

          if rel_device_num == 0:
            update_ops.extend(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope))
            assert not self.variable_mgr.staging_delta_ops

    enqueue_ops = []
    if gpu_compute_stage_ops:
      enqueue_ops.append(tf.group(*gpu_compute_stage_ops,
                                  name='gpu_compute_stage_ops'))
    assert not self.variable_mgr.supports_staged_vars()
    assert not gpu_grad_stage_ops

    fetches = self._build_fetches(global_step, logits, losses, grads,
                                  enqueue_ops, update_ops)
    return (None, enqueue_ops, fetches)

  # def build_graph(self, reader, model, batch_size, regularization_penalty):
  #   """Creates the Tensorflow graph.

  #   This will only be called once in the life of
  #   a training model, because after the graph is created the model will be
  #   restored from a meta graph file rather than being recreated.

  #   Args:
  #     reader: the input class.
  #     model: The core model.
  #     batch_size: How many examples to process at a time.
  #     regularization_penalty: How much weight to give the regularization loss
  #                             compared to the label loss.
  #   """
  #   with tf.device(self.global_step_device):
  #     global_step = tf.train.get_or_create_global_step()

  #   with tf.name_scope("input"):
  #     input_processing_info = InputProcessingInfo(
  #         input_producer_op=None,
  #         input_producer_stages=None,
  #         multi_device_iterator_input=None)
  #     input_processing_info = input_processing_info._replace(
  #       multi_device_iterator_input=reader.input_fn().get_next())

  #   logits, losses, gradients = [], [], []
  #   gpu_compute_stage_ops = []
  #   gpu_grad_stage_ops = []

  #   for device_num in range(len(self.devices)):
  #     with tf.name_scope('tower_{}'.format(device_num)) as name_scope, (
  #       self.variable_mgr.create_outer_variable_scope(device_num)):

  #         results = self.add_forward_pass_and_gradients(
  #             device_num, device_num, input_processing_info,
  #             gpu_compute_stage_ops, gpu_grad_stage_ops)

  #         losses.append(results['loss'])
  #         logits.append(results['logits'])
  #         gradients.append(results['gradvars'])

  #         if device_num == 0:
  #           update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

  #   fetches = self._build_fetches(global_step, logits, losses, gradients,
  #                                 update_ops)
  #   return fetches


  # def _build_fetches(self, global_step, all_logits, losses, device_grads,
  #                    update_ops):
  #   """Complete construction of model graph, populating the fetches map."""
  #   fetches = {}

  #   apply_gradient_devices, gradient_state = (
  #       self.variable_mgr.preprocess_device_grads(device_grads))

  #   # global_step is shared by all workers, and so every iteration
  #   # global_step is incremented by num_workers.
  #   if FLAGS.compute_lr_on_cpu:
  #     with tf.device(self.cpu_device):
  #       learning_rate = LearningRate(
  #         global_step, self.batch_size).get_learning_rate()

  #   training_ops = []
  #   for d, device in enumerate(apply_gradient_devices):
  #     with tf.device(device):
  #       with tf.name_scope('average_loss'):
  #         average_loss = tf.reduce_mean(losses)
  #       with tf.name_scope('get_gradients_to_apply'):
  #         avg_grads = self.variable_mgr.get_gradients_to_apply(
  #           d, gradient_state)

  #       if not FLAGS.compute_lr_on_cpu:
  #         # We compute the learning rate once for each device in
  #         # `apply_gradient_devices`.
  #         learning_rate = LearningRate(
  #           global_step, self.batch_size).get_learning_rate()
  #       if FLAGS.gradient_clip:
  #         with tf.name_scope('clip_gradients'):
  #           clipped_grads = [(tf.clip_by_value(grad, -FLAGS.gradient_clip,
  #                                              +FLAGS.gradient_clip), var)
  #                            for grad, var in avg_grads]
  #       else:
  #         clipped_grads = avg_grads

  #       learning_rate = tf.identity(learning_rate, name='learning_rate_tensor')
  #       opt = Optimizer(learning_rate).get_optimizer()

  #       # TODO: remove auto loss scale
  #       loss_scale_params = variable_mgr_util.AutoLossScaleParams(
  #           enable_auto_loss_scale=False,
  #           loss_scale=False,
  #           loss_scale_normal_steps=False,
  #           inc_loss_scale_every_n=0,
  #           is_chief=not self.job_name or self.task_index == 0)

  #       with tf.name_scope('append_apply_gradient_ops'):
  #         self.variable_mgr.append_apply_gradients_ops(
  #             gradient_state, opt, clipped_grads, training_ops,
  #             loss_scale_params)

  #   train_op = tf.group(*(training_ops + update_ops), name='train_ops_group')

  #   with tf.device(self.cpu_device):
  #     if self.task_index == 0 and FLAGS.summary_verbosity >= 1:
  #       tf.summary.scalar('learning_rate', learning_rate)
  #       tf.summary.scalar('loss', average_loss)
  #       if FLAGS.summary_verbosity >= 2:
  #         self.gradient_histogram_summary(avg_grads)

  #       if FLAGS.summary_verbosity >= 3:
  #         for grad, var in avg_grads:
  #           if grad is not None:
  #             tf.summary.histogram(var.op.name + '/gradients', grad)
  #         for var in tf.trainable_variables():
  #           tf.summary.histogram(var.op.name, var)

  #   fetches['global_step'] = tf.train.get_global_step()
  #   fetches['train_op'] = train_op
  #   fetches['loss'] = average_loss
  #   fetches['learning_rate'] = learning_rate
  #   return fetches



  def _build_fetches(self, global_step, all_logits, losses, device_grads,
                     enqueue_ops, update_ops):
    """Complete construction of model graph, populating the fetches map."""
    fetches = {}
    if enqueue_ops:
      fetches['enqueue_ops'] = enqueue_ops

    apply_gradient_devices, gradient_state = (
        self.variable_mgr.preprocess_device_grads(device_grads))

    # TODO(reedwm): Greatly simplify the learning rate code.
    if (self.variable_update == 'horovod' or
        self.variable_update == 'collective_all_reduce'):
      # Each worker independently increments global_step.
      examples_per_step = self.batch_size * self.num_workers
    else:
      # global_step is shared by all workers, and so every iteration
      # global_step is incremented by num_workers.
      examples_per_step = self.batch_size
    if FLAGS.compute_lr_on_cpu:
      with tf.device(self.cpu_device):
        learning_rate = LearningRate(
          global_step, self.batch_size).get_learning_rate()

    training_ops = []
    for d, device in enumerate(apply_gradient_devices):
      with tf.device(device):
        with tf.name_scope('average_loss'):
          average_loss = tf.reduce_mean(losses)
        with tf.name_scope('get_gradients_to_apply'):
          avg_grads = self.variable_mgr.get_gradients_to_apply(d,
                                                               gradient_state)

        if not FLAGS.compute_lr_on_cpu:
          # We compute the learning rate once for each device in
          # `apply_gradient_devices`.
          learning_rate = LearningRate(
            global_step, self.batch_size).get_learning_rate()

        gradient_clip = FLAGS.gradient_clip
        if gradient_clip is not None:
          with tf.name_scope('clip_gradients'):
            clipped_grads = [(tf.clip_by_value(grad, -gradient_clip,
                                               +gradient_clip), var)
                             for grad, var in avg_grads]
        else:
          clipped_grads = avg_grads

        learning_rate = tf.identity(learning_rate, name='learning_rate_tensor')
        opt = Optimizer(learning_rate).get_optimizer()
        # TODO: remove loss_scame params
        loss_scale_params = variable_mgr_util.AutoLossScaleParams(
            enable_auto_loss_scale=self.enable_auto_loss_scale,
            loss_scale=False,
            loss_scale_normal_steps=0,
            inc_loss_scale_every_n=0,
            is_chief=not self.job_name or self.task_index == 0)

        with tf.name_scope('append_apply_gradient_ops'):
          self.variable_mgr.append_apply_gradients_ops(
              gradient_state, opt, clipped_grads, training_ops,
              loss_scale_params)
    train_op = tf.group(*(training_ops + update_ops), name='train_ops_group')

    with tf.device(self.cpu_device):
      if self.task_index == 0 and FLAGS.summary_verbosity >= 1:
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('loss', average_loss)

        if FLAGS.summary_verbosity >= 2:
          self.gradient_histogram_summary(avg_grads)

        if FLAGS.summary_verbosity >= 3:
          for grad, var in avg_grads:
            if grad is not None:
              tf.summary.histogram(var.op.name + '/gradients', grad)
          for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    fetches['train_op'] = train_op
    fetches['loss'] = average_loss
    fetches['learning_rate'] = learning_rate
    return fetches


  def gradient_histogram_summary(self, avg_grads):
    """Create histogram of log values of all non-zero gradients."""
    with tf.name_scope('log_gradients_summary'):
      all_grads = []
      for grad, _ in avg_grads:
        all_grads.append(tf.reshape(grad, [-1]))
      grads = tf.abs(tf.concat(all_grads, 0))
      # exclude grads with zero values.
      indices_for_non_zero_grads = tf.where(tf.not_equal(grads, 0))
      log_grads = tf.reshape(
          tf.log(tf.gather(grads, indices_for_non_zero_grads)), [-1])
      tf.summary.histogram('log_gradients', log_grads)


  def add_sync_queues_and_barrier(self, name_prefix, enqueue_after_list):
    """Adds ops to enqueue on all worker queues.

    Args:
      name_prefix: prefixed for the shared_name of ops.
      enqueue_after_list: control dependency from ops.

    Returns:
      An op that should be used as control dependency before starting next step.
    """
    self.sync_queue_counter += 1
    with tf.device(self.sync_queue_devices[(
        self.sync_queue_counter % len(self.sync_queue_devices))]):
      sync_queues = [
          tf.FIFOQueue(self.num_workers, [tf.bool], shapes=[[]],
                       shared_name='%s%s' % (name_prefix, i))
          for i in range(self.num_workers)]
      queue_ops = []
      # For each other worker, add an entry in a queue, signaling that it can
      # finish this step.
      token = tf.constant(False)
      with tf.control_dependencies(enqueue_after_list):
        for i, q in enumerate(sync_queues):
          if i == self.task_index:
            queue_ops.append(tf.no_op())
          else:
            queue_ops.append(q.enqueue(token))

      # Drain tokens off queue for this worker, one for each other worker.
      queue_ops.append(
          sync_queues[self.task_index].dequeue_many(len(sync_queues) - 1))

      return tf.group(*queue_ops)


  def run(self):
    """Performs training on the currently defined Tensorflow graph.
    """
    # reset the training directory if start_new_model is True
    if self.is_master and self.start_new_model and exists(self.train_dir):
      self.remove_training_directory(self.train_dir)

    # save the parameters in json formet in the training directory
    model_flags_dict = FLAGS.to_json()
    log_folder = '{}_logs'.format(self.train_dir)
    flags_json_path = join(log_folder, "model_flags.json")
    if not exists(flags_json_path):
      with open(flags_json_path, "w") as fout:
        fout.write(model_flags_dict)

    if self.job_name == 'ps':
      log_fn('Running parameter server %s' % self.task_index)
      self.cluster_manager.join_server()
      return

    # For distributed_all_reduce with multiple workers, drive
    # from a separate controller process.
    if FLAGS.variable_update == 'distributed_all_reduce':
      if self.job_name == 'worker':
        logging.info('Starting worker {}'.format(self.task_index))
        self.cluster_manager.join_server()
        return
      elif FLAGS.job_name and FLAGS.job_name != 'controller':
        raise ValueError('unrecognized job name: {}'.format(FLAGS.job_name))

    with tf.Graph().as_default():
      self._run_training()


  def _run_training(self):

    # meta_filename = self.get_meta_filename()
    # if meta_filename:
    #   # TODO: fix it
    #   saver = self.recover_model(meta_filename)

    # if not meta_filename:
    #   fetches = self.build_graph(
    #               reader=self.reader,
    #               model=self.model,
    #               batch_size=self.batch_size,
    #               regularization_penalty=FLAGS.regularization_penalty)

    #   saver = tf.train.Saver(max_to_keep=0)

    if self.variable_update == 'distributed_all_reduce':
      self.single_session = True
      (_, enqueue_ops, fetches) = (
          self._build_model_single_session())
    else:
      self.single_session = False
      (_, enqueue_ops, fetches) = self._build_model()

    fetches_list = nest.flatten(list(fetches.values()))
    main_fetch_group = tf.group(*fetches_list, name='main_fetch_group')
    execution_barrier = None
    if (not self.single_session and self.job_name and
        not FLAGS.cross_replica_sync):
      execution_barrier = self.add_sync_queues_and_barrier(
          'execution_barrier_', [])

    global_step = tf.train.get_global_step()
    with tf.device(self.global_step_device), tf.name_scope('inc_global_step'):
      with tf.control_dependencies([main_fetch_group]):
        fetches['global_step'] = global_step.assign_add(1)

    if ((not self.single_session) and (not self.distributed_collective) and
        self.job_name and FLAGS.cross_replica_sync):
      # Block all replicas until all replicas are ready for next step.
      fetches['sync_queues'] = self.add_sync_queues_and_barrier(
          'sync_queues_step_end_', [main_fetch_group])

    with tf.name_scope('local_variable_initialization'):
      local_var_init_op = tf.local_variables_initializer()
    table_init_ops = tf.tables_initializer()

    variable_manager_init_ops = [local_var_init_op]
    if table_init_ops:
      variable_manager_init_ops.extend([table_init_ops])
    with tf.control_dependencies([local_var_init_op]):
      variable_manager_init_ops.extend(self.variable_mgr.get_post_init_ops())
    if ((not self.single_session) and (not self.distributed_collective) and
        self.job_name and FLAGS.cross_replica_sync):
      # Ensure all workers execute variable_manager_init_ops before they start
      # executing the model.
      variable_manager_init_ops.append(
          self.add_sync_queues_and_barrier('init_ops_end_',
                                           variable_manager_init_ops))
    local_var_init_op_group = tf.group(*variable_manager_init_ops,
                                       name='local_var_init_op_group')
    summary_op = tf.summary.merge_all()

    logging.info('Initializing graph')
    summary_writer = None
    if (self.is_master and FLAGS.summary_verbosity and FLAGS.train_dir and
        FLAGS.save_summaries_steps > 0):
      summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                             tf.get_default_graph())

    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    if self.is_master:
      saver = tf.train.Saver(
          self.variable_mgr.savable_variables(),
          save_relative_paths=True,
          max_to_keep=0)
    else:
      saver = None
    ready_for_local_init_op = None
    if self.job_name and not (self.single_session or
                              self.distributed_collective):
      # In distributed mode, we don't want to run local_var_init_op_group until
      # the global variables are initialized, because local_var_init_op_group
      # may use global variables (such as in distributed replicated mode). We
      # don't set this in non-distributed mode, because in non-distributed mode,
      # local_var_init_op_group may itself initialize global variables (such as
      # in replicated mode).
      ready_for_local_init_op = tf.report_uninitialized_variables(
          tf.global_variables())

    if FLAGS.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      bcast_global_variables_op = hvd.broadcast_global_variables(0)
    else:
      bcast_global_variables_op = None

    if self.variable_update == 'collective_all_reduce':
      # It doesn't matter what this collective_graph_key value is,
      # so long as it's > 0 and the same at every worker.
      init_run_options = tf.RunOptions()
      init_run_options.experimental.collective_graph_key = 6
    else:
      init_run_options = tf.RunOptions()

    scaffold = tf.train.Scaffold(
      saver=saver,
      ready_for_local_init_op=ready_for_local_init_op,
      local_init_op=local_var_init_op_group,
      summary_op=None)

    hooks = [
      tf.train.NanTensorHook(fetches['loss']),
      tf.train.StopAtStepHook(num_steps=self.max_steps)]

    # For the purpose of Supervisor, all Horovod workers are 'chiefs',
    # since we want session to be initialized symmetrically on all the
    # workers.
    is_chief = self.is_master or (self.variable_update == 'horovod'
                            or self.distributed_collective)
    logging.info('is_master {}'.format(self.is_master))
    logging.info('is_chief {}'.format(is_chief))

    session_args = dict(
      is_chief=is_chief,
      scaffold=scaffold,
      checkpoint_dir=FLAGS.train_dir,
      hooks=hooks,
      save_checkpoint_steps=FLAGS.save_checkpoint_steps,
      save_summaries_steps=10,
      save_summaries_secs=None,
      log_step_count_steps=0,
      config=create_config_proto())

    logging.info("Start training")
    with tf.train.MonitoredTrainingSession(**session_args) as sess:

      # if FLAGS.profiler:
      #   profiler = tf.profiler.Profiler(sess.graph)

      if bcast_global_variables_op:
        sess.run(bcast_global_variables_op)
      if enqueue_ops:
        for i in range(len(enqueue_ops)):
          sess.run(graph_info.enqueue_ops[:(i + 1)])
      # self.init_global_step = sess.run(global_step)
      # if self.job_name and not self.params.cross_replica_sync:
      #   # TODO(zhengxq): Do we need to use a global step watcher at all?
      #   global_step_watcher = GlobalStepWatcher(
      #       sess, graph_info.global_step,
      #       self.num_workers * self.num_warmup_batches +
      #       self.init_global_step,
      #       self.num_workers * (self.num_warmup_batches + self.num_batches) - 1)
      #   global_step_watcher.start()
      # else:
      #   global_step_watcher = None

      # if self.params.debugger is not None:
      #   if self.params.debugger == 'cli':
      #     log_fn('The CLI TensorFlow debugger will be used.')
      #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      #   else:
      #     log_fn('The TensorBoard debugger plugin will be used.')
      #     sess = tf_debug.TensorBoardDebugWrapperSession(sess,
      #                                                    self.params.debugger)

      results = {'global_step': 0}
      while not sess.should_stop():

        make_profile = False
        profile_args = {}

        # if FLAGS.profiler and results['global_step'] % 1000 == 0:
        #   make_profile = True
        #   run_meta = tf.RunMetadata()
        #   profile_args = {
        #     'options': tf.RunOptions(
        #       trace_level=tf.RunOptions.FULL_TRACE),
        #     'run_metadata': run_meta
        #   }

        # if gradients_norm != 0:
        #   fetches['gradients_norm'] = gradients_norm

        batch_start_time = time.time()
        results = sess.run(fetches, **profile_args)
        seconds_per_batch = time.time() - batch_start_time
        examples_per_second = self.batch_size / seconds_per_batch

        # if FLAGS.gradients['compute_hessian'] and results['global_step'] != 0 and \
        #    results['global_step'] % FLAGS.gradients['hessian_every_n_step'] == 0:
        #   compute_hessian_and_summary(sess, summary_writer,
        #                               results['global_step'])

        # if make_profile and FLAGS.profiler:
        #   profiler.add_step(results['global_step'], run_meta)

        #   # Profile the parameters of your model.
        #   profiler.profile_name_scope(options=(tf.profiler.ProfileOptionBuilder
        #       .trainable_variables_parameter()))

        #   # Or profile the timing of your model operations.
        #   opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
        #   profiler.profile_operations(options=opts)

        #   # Or you can generate a timeline:
        #   opts = (tf.profiler.ProfileOptionBuilder(
        #           tf.profiler.ProfileOptionBuilder.time_and_memory())
        #           .with_step(results['global_step'])
        #           .with_timeline_output('~/profile.logs').build())
        #   profiler.profile_graph(options=opts)

        to_print = results['global_step'] % FLAGS.frequency_log_steps == 0
        if (self.is_master and to_print) or results['global_step'] == 1:
          epoch = ((results['global_step'] * self.batch_size)
            / self.reader.n_train_files)

          message = MessageBuilder()
          message.add("epoch", epoch, format="4.2f")
          message.add("step", results['global_step'], width=5, format=".0f")
          message.add("lr", results['learning_rate'], format=".6f")
          message.add("loss", results['loss'], format=".4f")
          if "YT8M" in self.reader.__class__.__name__:
            gap = eval_util.calculate_gap(
              results['logits'], results['labels'])
            message.add("gap", gap, format=".3f")
          message.add("imgs/sec", examples_per_second, width=5, format=".0f")
          if FLAGS.gradients['perturbed_gradients']:
            message.add("grad norm", results['gradients_norm'], format=".4f")
          logging.info(message.get_message())

      # End training
      logging.info("{}: Done training -- epoch limit reached.".format(
        task_as_string(self.task)))
      # if FLAGS.profiler:
      #   profiler.advise()
    logging.info("{}: Exited training loop.".format(task_as_string(self.task)))


  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info(("Train dir already exist and start_new_model "
                    "set to True. To restart model from scratch, "
                    "delete the directory."))
      gfile.DeleteRecursively(train_dir)
      # sys.exit()
    except:
      logging.error("Failed to delete directory {} when starting a new "
        "model. Please delete it manually and try again.".format(train_dir))
      sys.exit()


  def get_meta_filename(self):
    if self.start_new_model:
      logging.info("Flag 'start_new_model' is set. Building a new "
        "model.")
      return None

    latest_checkpoint = tf.train.latest_checkpoint(self.train_dir)
    if not latest_checkpoint:
      logging.info("No checkpoint file found. Building a new model.")
      return None

    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("No meta graph file found. Building a new model.")
      return None
    else:
      return meta_filename


  def recover_model(self, meta_filename):
    logging.info("Restoring from meta graph file {}".format(meta_filename))
    return tf.train.import_meta_graph(meta_filename,
      clear_devices=FLAGS.clear_devices)



def set_logging_verbosity(verbosity):
  v = {
    'DEBUG': 10,
    'ERROR': 40,
    'FATAL': 50,
    'INFO': 20,
    'WARN': 30
  }[verbosity]
  logging.set_verbosity(v)

def main():

  # Sets up the environment that BenchmarkCNN should run in.
  setup()

  # Setup logging & log the version.
  set_logging_verbosity(FLAGS.logging_verbosity)
  logging.info("Tensorflow version: {}.".format(tf.__version__))
  logging.info("Hostname: {}.".format(socket.gethostname()))

  # print FLAGS parameters
  pp = pprint.PrettyPrinter(indent=2, compact=True)
  logging.info(pp.pformat(FLAGS.values()))

  # run Trainer
  trainer = Trainer()
  trainer.run()


if __name__ == "__main__":
  main()
