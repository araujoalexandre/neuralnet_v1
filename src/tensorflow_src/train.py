
import time
import pprint
import socket
import logging
from os.path import join
from os.path import exists

import utils as global_utils
from . import utils
from .models import model_config
from .dataset.readers import readers_config
from .train_utils import losses
from .train_utils.learning_rate import get_learning_rate
from .train_utils import variable_mgr, variable_mgr_util

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from tensorflow.io import gfile
from tensorflow.python.util import nest


def maybe_compile(computation, params):
  if params and params.xla_compile:
    return tf.xla.experimental.compile(computation)
  else:
    return computation()


def setup(params):
  """Sets up the environment that BenchmarkCNN should run in.

  Returns:
    A potentially modified params.
  Raises:
    ValueError: invalid parames combinations.
  """
  # Set up environment variables before doing any other global initialization to
  # make sure it uses the appropriate environment variables.
  utils.set_default_param_values_and_env_vars(params)
  # platforms_util.initialize(params, create_config_proto(params))
  if params.job_name:
    # Create a dummy session to initialize TF global variables using the input
    # params. Otherwise, ListDevices function may create global devices using
    # the default config instead of using the user provided config.
    #
    # TODO(hinsu): Find a way to achieve the same for distributed benchmark. It
    # is not legal to create distributed session after local session. It is also
    # not possible to create distributed session here as that results in
    # multiple creation of ClusterManager and Server.
    with tf.Session(config=utils.create_config_proto(params)) as sess:
      del sess


def params_sanity_checks(params):
  """Checks if parameters are coherent, raise Error otherwise.
  """
  if (params.device == 'cpu' and params.data_format == 'NCHW' and
      not params.mkl):
    raise ValueError('device=cpu requires that data_format=NHWC')

  if (params.use_fp16 and params.fp16_vars and
      'replicated' in params.variable_update and
      params.all_reduce_spec and 'nccl' in params.all_reduce_spec):
    raise ValueError('fp16 variables are not supported with NCCL')
  if (params.use_fp16 and params.fp16_vars and
      params.gradient_repacking):
    raise ValueError('--fp16_vars cannot be used with --gradient_repacking')

  if params.variable_update == 'horovod' and params.num_gpus > 1:
    raise ValueError('Horovod benchmarks require num_gpus=1 on each worker')

  if params.variable_update == 'horovod' and params.job_name:
    raise ValueError('job_name should not be specified for Horovod.')

  if params.use_fp16 and params.fp16_enable_auto_loss_scale:
    if params.all_reduce_spec and 'nccl' in params.all_reduce_spec:
      raise ValueError('Automatic loss scaling is not supported with NCCL.')
    if params.variable_update not in ('parameter_server', 'replicated',
                                           'independent'):
      raise ValueError('Automatic loss scaling is not supported with '
                       'variable_update=%s.' % params.variable_update)

  # controller is used for distributed_all_reduce with > 1 worker.
  use_controller = (
      params.variable_update == 'distributed_all_reduce' and
      params.job_name)
  if use_controller and not self.params.controller_host:
    raise ValueError('When variable_update==distributed_all_reduce '
                     'controller_host must also be specified.')



def generate_tfprof_profile(profiler, tfprof_file):
  """Generates a tfprof profile, writing it to a file and printing top ops.

  Args:
    profiler: A tf.profiler.Profiler. `profiler.add_step` must have already been
      called.
    tfprof_file: The filename to write the ProfileProto to.
  """
  profile_proto = profiler.serialize_to_string()
  logging.info('Dumping ProfileProto to %s' % tfprof_file)
  with gfile.Open(tfprof_file, 'wb') as f:
    f.write(profile_proto)

  # Print out the execution times of the top operations. Note this
  # information can also be obtained with the dumped ProfileProto, but
  # printing it means tfprof doesn't have to be used if all the user wants
  # is the top ops.
  options = tf.profiler.ProfileOptionBuilder.time_and_memory()
  options['max_depth'] = 20
  options['order_by'] = 'accelerator_micros'
  profiler.profile_operations(options)


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


def get_optimizer(optimizer, params, learning_rate):
  """Returns the optimizer that should be used based on params."""
  if optimizer == 'momentum':
    opt = tf_v1.train.MomentumOptimizer(
        learning_rate, params['momentum'], use_nesterov=True)
  elif optimizer == 'sgd':
    opt = tf_v1.train.GradientDescentOptimizer(learning_rate)
  elif optimizer == 'rmsprop':
    opt = tf_v1.train.RMSPropOptimizer(
        learning_rate,
        params['decay'],
        momentum=params['momentum'],
        epsilon=params['epsilon'])
  elif optimizer == 'adam':
    opt = tf_v1.train.AdamOptimizer(learning_rate, params['beta1'],
                                 params['beta2'], params['epsilon'])
  else:
    raise ValueError('Optimizer "{}" was not recognized'.
                     format(params.optimizer))
  return opt


class Trainer:
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, params):
    """Creates a Trainer.
    """
    self.params = params
    params_sanity_checks(params)

    # Sets up the environment that Trainer should run in.
    setup(params)

    # Setup logging & log the version.
    global_utils.setup_logging(params.logging_verbosity)
    logging.info("Tensorflow version: {}.".format(tf.__version__))
    logging.info("Hostname: {}.".format(socket.gethostname()))

    # print self.params parameters
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(params.values()))

    self.job_name = self.params.job_name  # "" for local training
    self.task_index = self.params.task_index
    self.is_master = (self.job_name in ('', 'worker') and self.task_index == 0)
    self.start_new_model = self.params.start_new_model
    self.train_dir = self.params.train_dir
    self.num_gpus = self.params.num_gpus
    self.variable_update = self.params.variable_update

    # create a mesage builder for logging
    self.message = global_utils.MessageBuilder()

    self.batch_size = self.params.batch_size * self.num_gpus
    if self.job_name:
      self.global_batch_size = self.batch_size * \
          (self.cluster.num_tasks('worker') + 1)
    else:
      self.global_batch_size = self.batch_size

    self.trace_filename = self.params.trace_file
    self.sync_queue_counter = 0

    # TODO: remove auto loss scale and check inf in grad
    self.enable_auto_loss_scale = (
        self.params.use_fp16 and self.params.fp16_enable_auto_loss_scale)
    self.loss_scale = None
    self.loss_scale_normal_steps = None

    # PS server is used for distributed jobs not using all-reduce.
    use_ps_server = self.job_name and \
      (self.variable_update != 'distributed_all_reduce' and \
       self.variable_update != 'collective_all_reduce')

    # collective_all_reduce doesn't need a controller or ps
    self.distributed_collective = (
        self.variable_update == 'collective_all_reduce' and
        self.job_name)

    self.local_parameter_device_flag = self.params.local_parameter_device
    if self.job_name:
      self.cluster_manager = platforms_util.get_cluster_manager(
          params, utils.create_config_proto(params))
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
      self.param_server_device = '/{}:0'.format(self.params.local_parameter_device)
      self.sync_queue_devices = [self.param_server_device]

    if self.cluster_manager:
      self.num_workers = self.cluster_manager.num_workers()
    elif self.params.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      self.num_workers = hvd.size()
    else:
      self.num_workers = 1
    self.num_ps = self.cluster_manager.num_ps() if self.cluster_manager else 0

    if self.num_workers > 1 and self.params.all_reduce_spec == 'nccl':
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

    if self.params.variable_update == 'parameter_server':
      if self.job_name:
        self.variable_mgr = \
            variable_mgr.VariableMgrDistributedFetchFromPS(self)
      else:
        self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
    elif self.variable_update == 'replicated':
      if self.job_name:
        raise ValueError('Invalid variable_update in distributed mode: %s' %
                         self.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
          self, self.params.all_reduce_spec,
          self.params.agg_small_grads_max_bytes,
          self.params.agg_small_grads_max_group,
          self.params.allreduce_merge_scope)
    elif self.variable_update == 'distributed_all_reduce':
      assert self.params.cross_replica_sync
      self.variable_mgr = variable_mgr.VariableMgrDistributedAllReduce(
          self, self.params.all_reduce_spec,
          ('worker' if self.num_workers > 1 else 'localhost'),
          self.num_workers, self.params.agg_small_grads_max_bytes,
          self.params.agg_small_grads_max_group,
          self.params.allreduce_merge_scope)
    elif self.params.variable_update == 'collective_all_reduce':
      assert self.params.cross_replica_sync
      self.variable_mgr = variable_mgr.VariableMgrCollectiveAllReduce(
          self, self.params.all_reduce_spec,
          self.num_workers, self.num_gpus, self.task_index,
          self.params.allreduce_merge_scope)
    elif self.variable_update == 'distributed_replicated':
      assert self.params.cross_replica_sync
      if not self.job_name:
        raise ValueError('Invalid variable_update in local mode: {}'.format(
                         self.variable_update))
      self.variable_mgr = variable_mgr.VariableMgrDistributedReplicated(self)
    elif self.params.variable_update in ('independent', 'horovod'):
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
      elif self.params.variable_update == 'collective_all_reduce':
        self.global_step_device = self.cpu_device
      else:
        self.global_step_device = '/job:worker/replica:0/task:0/cpu:0'
    else:
      self.global_step_device = self.cpu_device


    self.enable_auto_loss_scale = False

    self.model = model_config.get_model_config(
        self.params.model, self.params.dataset, self.params)
    self.reader = readers_config[self.params.dataset](
      self.params, self.batch_size, self.raw_devices,
      self.cpu_device, is_training=True)

    # define the number of steps
    self.num_steps_by_epoch = self.reader.n_train_files / self.global_batch_size
    self.max_steps = self.params.num_epochs * self.num_steps_by_epoch


  def add_forward_pass_and_gradients(self,
                                     rel_device_num,
                                     abs_device_num,
                                     multi_device_iterator_input):
    """Add ops for forward-pass and gradient computations."""
    n_classes = self.reader.n_classes
    is_training = True
    input_list = multi_device_iterator_input[rel_device_num]

    def forward_pass_and_gradients():
      """Builds forward pass and gradient computation network.

      Returns:
        outputs: A list of tensors depending on different modes.
      """
      build_network_result = self.model.build_network(
        input_list, is_training, n_classes)
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
          if self.params.single_l2_loss_op:
            reshaped_params = [tf.reshape(p, (-1,)) for p in filtered_params]
            l2_loss = tf.nn.l2_loss(tf.concat(reshaped_params, axis=0))
          else:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in filtered_params])
      weight_decay = self.params.weight_decay
      if (weight_decay is not None and weight_decay != 0. and
          l2_loss is not None):
        total_loss += len(self.devices) * weight_decay * l2_loss

      aggmeth = tf.AggregationMethod.DEFAULT
      grads = tf.gradients(total_loss, params, aggregation_method=aggmeth)
      if self.params.variable_update == 'horovod':
        import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
        if self.params.horovod_device:
          horovod_device = '/%s:0' % self.params.horovod_device
        else:
          horovod_device = ''
        # All-reduce gradients using Horovod.
        grads = [hvd.allreduce(grad, average=False, device_dense=horovod_device)
                 for grad in grads]

      results = {}
      results['logits'] = logits
      results['loss'] = total_loss
      results['grads'] = grads
      param_refs = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num, writable=True)
      results['gradvars'] = list(zip(results['grads'], param_refs))
      return results

    with tf.device(self.devices[rel_device_num]):
      return maybe_compile(forward_pass_and_gradients, self.params)


  def _maybe_initialize_fp16(self):
    """Initialize fp16 settings."""
    if self.params.use_fp16:
      init_loss_scale_val = float(self.params.fp16_loss_scale or
                                  self.model.get_fp16_loss_scale())
      self.loss_scale = None
      self.loss_scale_normal_steps = None
      if self.enable_auto_loss_scale or init_loss_scale_val != 1:
        self.loss_scale = tf.get_variable(
            name='loss_scale',
            initializer=init_loss_scale_val,
            dtype=tf.float32,
            trainable=False)
      if self.enable_auto_loss_scale:
        self.loss_scale_normal_steps = tf.get_variable(
            name='loss_scale_normal_steps', initializer=0, trainable=False)


  def _build_model(self):
    """Build the TensorFlow graph."""
    # Adjust seed so different workers start read different input files.
    if self.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      seed_adjustment = hvd.rank()
    else:
      seed_adjustment = 0
    tf_v1.set_random_seed(self.params.tf_random_seed + seed_adjustment)
    np.random.seed(4321 + seed_adjustment)
    is_training = True

    losses, logits, grads = [], [], []

    with tf.device(self.global_step_device):
      global_step = tf_v1.train.get_or_create_global_step()
      self._maybe_initialize_fp16()

    # Build the processing and model for the worker.
    with tf.name_scope("input"):
      # TODO: add shift_ratio = 0
      multi_device_iterator_input = self.reader.input_fn().get_next()

    update_ops = None
    for device_num in range(len(self.devices)):
      with tf.name_scope('tower_{}'.format(device_num)) as name_scope, (
          self.variable_mgr.create_outer_variable_scope(device_num)):
        results = self.add_forward_pass_and_gradients(
            device_num, device_num, multi_device_iterator_input)

        losses.append(results['loss'])
        logits.append(results['logits'])
        grads.append(results['gradvars'])

        if device_num == 0:
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
          # TODO: remove attributes
          assert not self.variable_mgr.staging_delta_ops

    fetches = self._build_fetches(global_step, losses, grads, update_ops)
    return fetches


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

    if self.params.random_seed != 0:
      tf.set_random_seed(self.params.random_seed)
      np.random.seed(self.params.random_seed + 1)
    is_training = True

    losses, logits, grads = [], [], []
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

      # TODO: add shift_ratio=(task_num / self.num_workers)
      with tf.name_scope("input"):
        multi_device_iterator_input = self.reader.input_fn().get_next()

      # Build the per-worker model replica.
      for rel_device_num in range(len(self.devices)):
        abs_device_num = task_num * len(self.devices) + rel_device_num
        name_scope = 'task_{}_tower_{}'.format(task_num, rel_device_num)
        with self.variable_mgr.create_outer_variable_scope(
            abs_device_num), tf.name_scope(name_scope) as name_scope:
          results = self.add_forward_pass_and_gradients(
              rel_device_num, abs_device_num,
              multi_device_iterator_input)

          losses.append(results['loss'])
          logits.append(results['logits'])
          grads.append(results['gradvars'])

          if rel_device_num == 0:
            update_ops.extend(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope))
            assert not self.variable_mgr.staging_delta_ops

    fetches = self._build_fetches(global_step, losses, grads, update_ops)
    return enqueue_ops, fetches


  def _build_fetches(self, global_step, losses, device_grads, update_ops):
    """Complete construction of model graph, populating the fetches map."""
    fetches = {}

    apply_gradient_devices, gradient_state = (
        self.variable_mgr.preprocess_device_grads(device_grads))

    # TODO: examples_per_step is is not used  
    if (self.variable_update == 'horovod' or
        self.variable_update == 'collective_all_reduce'):
      # Each worker independently increments global_step.
      examples_per_step = self.batch_size * self.num_workers
    else:
      # global_step is shared by all workers, and so every iteration
      # global_step is incremented by num_workers.
      examples_per_step = self.batch_size
    if self.params.compute_lr_on_cpu:
      with tf.device(self.cpu_device):
        learning_rate = get_learning_rate(
          self.params, global_step, self.num_steps_by_epoch, self.model)

    training_ops = []
    for d, device in enumerate(apply_gradient_devices):
      with tf.device(device):
        with tf.name_scope('average_loss'):
          average_loss = tf.reduce_mean(losses)
        with tf.name_scope('get_gradients_to_apply'):
          avg_grads = self.variable_mgr.get_gradients_to_apply(
            d, gradient_state)

        if not self.params.compute_lr_on_cpu:
          # We compute the learning rate once for each device in
          # `apply_gradient_devices`.
          learning_rate = get_learning_rate(
            self.params, global_step, self.num_steps_by_epoch, self.model)

        gradient_clip = self.params.gradient_clip
        if gradient_clip:
          with tf.name_scope('clip_gradients'):
            clipped_grads = [(tf.clip_by_value(grad, -gradient_clip,
                                               +gradient_clip), var)
                             for grad, var in avg_grads]
        else:
          clipped_grads = avg_grads

        learning_rate = tf.identity(learning_rate, name='learning_rate_tensor')
        self.opt = get_optimizer(self.params.optimizer,
          self.params.optimizer_params, learning_rate)
        # TODO: remove loss_scame params
        loss_scale_params = variable_mgr_util.AutoLossScaleParams(
            enable_auto_loss_scale=self.enable_auto_loss_scale,
            loss_scale=False,
            loss_scale_normal_steps=0,
            inc_loss_scale_every_n=0,
            is_chief=not self.job_name or self.task_index == 0)

        with tf.name_scope('append_apply_gradient_ops'):
          self.variable_mgr.append_apply_gradients_ops(
              gradient_state, self.opt, clipped_grads, training_ops,
              loss_scale_params)
    train_op = tf.group(*(training_ops + update_ops), name='train_ops_group')

    with tf.device(self.cpu_device):
      if self.task_index == 0 and self.params.summary_verbosity >= 1:
        tf_v1.summary.scalar('learning_rate', learning_rate)
        tf_v1.summary.scalar('loss', average_loss)

        if self.params.summary_verbosity >= 2:
          self.gradient_histogram_summary(avg_grads)

        if self.params.summary_verbosity >= 3:
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
      global_utils.remove_training_directory(self.train_dir)

    # save the parameters in json format in the training directory
    log_folder = '{}_logs'.format(self.train_dir)
    flags_json_path = join(log_folder, "model_flags.json")
    if not exists(flags_json_path):
      with open(flags_json_path, "w") as fout:
        fout.write(self.params.to_json())

    if self.job_name == 'ps':
      logging.info('Running parameter server {}'.format(self.task_index))
      self.cluster_manager.join_server()
      return

    # For distributed_all_reduce with multiple workers, drive
    # from a separate controller process.
    if self.params.variable_update == 'distributed_all_reduce':
      if self.job_name == 'worker':
        logging.info('Starting worker {}'.format(self.task_index))
        self.cluster_manager.join_server()
        return
      elif self.params.job_name and self.params.job_name != 'controller':
        raise ValueError('unrecognized job name: {}'.format(self.params.job_name))

    with tf.Graph().as_default():
      self._run_training()


  def _run_training(self):

    if self.variable_update == 'distributed_all_reduce':
      self.single_session = True
      fetches = self._build_model_single_session()
    else:
      self.single_session = False
      fetches = self._build_model()

    fetches_list = nest.flatten(list(fetches.values()))
    main_fetch_group = tf.group(*fetches_list, name='main_fetch_group')
    execution_barrier = None
    if (not self.single_session and self.job_name and
        not self.params.cross_replica_sync):
      execution_barrier = self.add_sync_queues_and_barrier(
          'execution_barrier_', [])

    global_step = tf_v1.train.get_global_step()
    with tf.device(self.global_step_device), tf.name_scope('inc_global_step'):
      with tf.control_dependencies([main_fetch_group]):
        fetches['global_step'] = global_step.assign_add(1)

    if ((not self.single_session) and (not self.distributed_collective) and
        self.job_name and self.params.cross_replica_sync):
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
        self.job_name and self.params.cross_replica_sync):
      # Ensure all workers execute variable_manager_init_ops before they start
      # executing the model.
      variable_manager_init_ops.append(
          self.add_sync_queues_and_barrier('init_ops_end_',
                                           variable_manager_init_ops))
    local_var_init_op_group = tf.group(*variable_manager_init_ops,
                                       name='local_var_init_op_group')
    summary_op = tf_v1.summary.merge_all()

    logging.info('Initializing graph')
    summary_writer = None
    if (self.is_master and self.params.summary_verbosity and \
        self.params.train_dir and self.params.save_summaries_steps > 0):
      summary_writer = tf_v1.summary.FileWriter(self.params.train_dir,
                                             tf.get_default_graph())

    if self.is_master:
      saver = tf_v1.train.Saver(
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

    # we need to set this in order to restore model with
    # MonitoredTrainingSession because when the train_dir is not empty
    # MonitoredTrainingSession does not run the init_op. 
    # TODO: test in distributed mode
    ready_for_local_init_op = tf.report_uninitialized_variables(
      self.variable_mgr.savable_variables())

    if self.params.variable_update == 'horovod':
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

    # TODO: We should run the summaries in the same thread as the training 
    # operations by passing in None for summary_op to avoid a summary_thread 
    # being started. Running summaries and training operations in parallel 
    # could run out of GPU memory.
    scaffold = tf_v1.train.Scaffold(
      saver=saver,
      ready_for_local_init_op=ready_for_local_init_op,
      local_init_op=local_var_init_op_group,
      summary_op=summary_op)

    hooks = [
      tf.estimator.NanTensorHook(fetches['loss']),
      tf.estimator.StopAtStepHook(num_steps=self.max_steps)]

    # For the purpose of Supervisor, all Horovod workers are 'chiefs',
    # since we want session to be initialized symmetrically on all the
    # workers.
    is_chief = self.is_master or (self.variable_update == 'horovod'
                            or self.distributed_collective)
    session_args = dict(
      is_chief=is_chief,
      scaffold=scaffold,
      checkpoint_dir=self.params.train_dir,
      hooks=hooks,
      save_checkpoint_steps=self.params.save_checkpoint_steps,
      save_summaries_steps=self.params.save_summaries_steps,
      log_step_count_steps=0,
      config=utils.create_config_proto(self.params))

    collective_graph_key = 7 if (
        self.params.variable_update == 'collective_all_reduce') else 0
    profiler = tf_v1.profiler.Profiler() if self.params.tfprof_file else None

    logging.info("Start training")
    with tf_v1.train.MonitoredTrainingSession(**session_args) as sess:

      if bcast_global_variables_op:
        sess.run(bcast_global_variables_op)
      self.init_global_step = sess.run(global_step)
      step = 0
      while not sess.should_stop():
        step = self._training(
          sess, step, fetches, profiler, collective_graph_key)
      logging.info("Done training -- epoch limit reached.")
      if self.params.tfprof_file:
        generate_tfprof_profile(profiler, self.params.tfprof_file)


  def _training(self, sess, step, fetches, profiler,
                collective_graph_key):

    should_profile = profiler and 0 <= step < 20
    need_options_and_metadata = (
        should_profile or collective_graph_key > 0 or
        (self.trace_filename and step == 0))
    if need_options_and_metadata:
      run_options = tf.RunOptions()
      if (self.trace_filename and step == 0) or should_profile:
        run_options.trace_level = tf.RunOptions.FULL_TRACE
      if collective_graph_key > 0:
        run_options.experimental.collective_graph_key = collective_graph_key
      run_metadata = tf.RunMetadata()
    else:
      run_options = None
      run_metadata = None

    batch_start_time = time.time()
    results = sess.run(fetches, options=run_options, run_metadata=run_metadata)
    seconds_per_batch = time.time() - batch_start_time
    examples_per_second = self.batch_size / seconds_per_batch
    step = results['global_step']

    to_print = step % self.params.frequency_log_steps == 0
    if (self.is_master and to_print) or step == 1:
      epoch = ((step * self.batch_size)
        / self.reader.n_train_files)
      self.message.add("epoch", epoch, format="4.2f")
      self.message.add("step", step, width=5, format=".0f")
      self.message.add("lr", results['learning_rate'], format=".6f")
      self.message.add("loss", results['loss'], format=".4f")
      self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")
      logging.info(self.message.get_message())

    if need_options_and_metadata:
      if should_profile:
        profiler.add_step(step, run_metadata)
      if trace_filename and step == -2:
        logging.info('Dumping trace to {}'.filename(trace_filename))
        trace_dir = os.path.dirname(trace_filename)
        if not gfile.exists(trace_dir):
          gfile.makedirs(trace_dir)
        with gfile.open(trace_filename, 'w') as trace_file:
          trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
    return step




