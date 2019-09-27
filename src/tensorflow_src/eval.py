
import json
import time
import os
import re
import socket
import pprint
import logging
from os.path import join
from os.path import exists

import utils as global_utils
from . import attacks
from . import utils
from .models import model, model_config
from .dataset.readers import readers_config
from .train_utils import variable_mgr, variable_mgr_util
from .utils import make_summary
from .dump_files import DumpFiles

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1



class Evaluator:
  """Evaluate a Tensorflow Model."""

  def __init__(self, params):

    self.params = params

    # Set up environment variables before doing any other global initialization to
    # make sure it uses the appropriate environment variables.
    utils.set_default_param_values_and_env_vars(params)

    # Setup logging & log the version.
    global_utils.setup_logging(params.logging_verbosity)
    logging.info("Tensorflow version: {}.".format(tf.__version__))
    logging.info("Hostname: {}.".format(socket.gethostname()))

    # print self.params parameters
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(params.values()))

    self.train_dir = self.params.train_dir
    self.logs_dir = "{}_logs".format(self.train_dir)
    if self.train_dir is None:
      raise ValueError('Trained model directory not specified')
    self.num_gpus = self.params.num_gpus
    self.variable_update = self.params.variable_update

    # create a mesage builder for logging
    self.message = global_utils.MessageBuilder()

    # class for dumping data
    if self.params.eval_under_attack:
      self.dump = DumpFiles(params)

    if self.params.num_gpus:
      self.batch_size = self.params.batch_size * self.num_gpus
    else:
      self.batch_size = self.params.batch_size

    if self.params.eval_under_attack:
      attack_method = self.params.attack_method
      attack_cls = getattr(attacks, attack_method, None)
      if attack_cls is None:
        raise ValueError("Attack is not recognized.")
      attack_config = getattr(self.params, attack_method)
      self.attack = attack_cls(batch_size=self.batch_size,
                               sample=self.params.attack_sample,
                               **attack_config)

    data_pattern = self.params.data_pattern
    self.dataset = re.findall("[a-z0-9]+", data_pattern.lower())[0]
    if data_pattern is "":
      raise IOError("'data_pattern' was not specified. "
        "Nothing to evaluate.")

    self.local_parameter_device_flag = self.params.local_parameter_device
    self.task_index = 0
    self.cluster_manager = None
    self.param_server_device = '/{}:0'.format(self.params.local_parameter_device)
    self.sync_queue_devices = [self.param_server_device]

    self.num_workers = 1

    # Device to use for ops that need to always run on the local worker's CPU.
    self.cpu_device = '/cpu:0'

    # Device to use for ops that need to always run on the local worker's
    # compute device, and never on a parameter server device.
    self.raw_devices = ['/gpu:{}'.format(i) for i in range(self.num_gpus)]

    if self.params.variable_update == 'parameter_server':
      self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
    elif self.variable_update == 'replicated':
      self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
          self, self.params.all_reduce_spec,
          self.params.agg_small_grads_max_bytes,
          self.params.agg_small_grads_max_group,
          self.params.allreduce_merge_scope)
    elif self.params.variable_update in 'independent':
      self.variable_mgr = variable_mgr.VariableMgrIndependent(self)
    else:
      raise ValueError(
          'Invalid variable_update in eval mode: {}'.format(
            self.variable_update))

    self.devices = self.variable_mgr.get_devices()

    # TODO: remove auto loss scale and check inf in grad
    self.enable_auto_loss_scale = False

    self.model = model_config.get_model_config(
        self.params.model, self.params.dataset, self.params)
    self.reader = readers_config[self.params.dataset](
      self.params, self.batch_size, self.raw_devices,
      self.cpu_device, is_training=False)

    # # define the number of steps
    # num_steps_by_epochs = self.reader.n_train_files / self.global_batch_size
    # self.max_steps = self.params.num_epochs * num_steps_by_epochs


  def run(self):
    """Run the benchmark task assigned to this process.
    """
    with tf.Graph().as_default():
      self._run_eval()
      logging.info("Done evaluation -- number of eval reached.")


  def _run_eval(self):
    """Evaluate a model every self.params.eval_interval_secs.

    Returns:
      Dictionary containing eval statistics. Currently returns an empty
      dictionary.

    Raises:
      ValueError: If self.params.train_dir is unspecified.
    """
    logging.info("Building evaluation graph")
    fetches = self._build_model()
    local_var_init_op = tf.local_variables_initializer()
    table_init_ops = tf.tables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    if table_init_ops:
      variable_mgr_init_ops.extend([table_init_ops])
    with tf.control_dependencies([local_var_init_op]):
      variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
    local_var_init_op_group = tf.group(*variable_mgr_init_ops)
    self.saver = tf_v1.train.Saver(self.variable_mgr.savable_variables())
    filename_suffix= "_{}_{}".format("eval",
                re.findall("[a-z0-9]+", self.params.data_pattern.lower())[0])
    self.summary_writer = tf_v1.summary.FileWriter(
      self.params.train_dir,
      filename_suffix=filename_suffix,
      graph=tf.get_default_graph())

    config = utils.create_config_proto(self.params)
    session = tf.Session(target='', config=config)

    if self.params.eval_under_attack:
      logging.info("Evaluation under attack:")
      with session as sess:
        sess.run(local_var_init_op_group)
        # pass session to attack class for Carlini Attack
        self.attack.sess = sess
        # Restores from bast checkpoint
        best_checkpoint, global_step = \
            global_utils.get_best_checkpoint(
              self.logs_dir, backend='tensorflow')
        self.saver.restore(sess, best_checkpoint)
        acc_val, acc_adv_val = self.eval_attack(sess, fetches)
      path = join(self.logs_dir, "attacks_score.txt")
      with open(path, 'a') as f:
        f.write("{}\n".format(self.params.attack_method))
        f.write("sample {}, {}\n".format(self.params.attack_sample,
                                       json.dumps(attack_config)))
        f.write("{:.5f}\t{:.5f}\n\n".format(acc_val, acc_adv_val))

    else:
      # those variables are updated in eval_loop
      self.best_global_step = None
      self.best_accuracy = None
      with session as sess:
        # if the evaluation is made during training, we don't know how many 
        # checkpoint we need to process
        if self.params.eval_during_training:
          last_global_step = 0
          while True:
            latest_checkpoint, global_step = global_utils.get_checkpoint(
              self.train_dir, last_global_step, backend='tensorflow')
            if latest_checkpoint is None or global_step == last_global_step:
              time.sleep(self.params.eval_interval_secs)
              continue
            else:
              logging.info(
                "Loading checkpoint for eval: {}".format(latest_checkpoint))
              # Restores from checkpoint
              self.saver.restore(sess, latest_checkpoint)
              sess.run(local_var_init_op_group)
              self.eval_loop(sess, fetches, global_step)
              last_global_step = global_step
        # if the evaluation is made after training, we look for all
        # checkpoints 
        else:
          ckpts = global_utils.get_list_checkpoints(
            self.train_dir, backend='tensorflow')
          # remove first checkpoint model.ckpt-0
          ckpts.pop(0)
          for ckpt in ckpts:
            logging.info(
              "Loading checkpoint for eval: {}".format(ckpt))
            global_step = global_utils.get_global_step_from_ckpt(ckpt)
            # Restores from checkpoint
            self.saver.restore(sess, ckpt)
            sess.run(local_var_init_op_group)
            self.eval_loop(sess, fetches, global_step)

      path = join(self.logs_dir, "best_accuracy.txt")
      with open(path, 'w') as f:
        f.write("{}\t{:.4f}\n".format(
          self.best_global_step, self.best_accuracy))



  def add_forward_pass_and_gradients(self,
                                     rel_device_num,
                                     abs_device_num,
                                     all_input_list,
                                     gpu_compute_stage_ops,
                                     gpu_grad_stage_ops):
    """Add ops for forward-pass and gradient computations."""
    input_list = all_inpout_list[rel_device_num]


    with tf.device(self.devices[rel_device_num]):
      logits = forward_pass_and_gradients(input_list)
      return logits


  def _build_model(self):
    """Build the TensorFlow graph."""
    tf.set_random_seed(self.params.tf_random_seed)
    np.random.seed(4321)
    n_classes = self.reader.n_classes
    is_training = False
    fetches = {}

    def forward_pass(input_list, return_loss=True):
      """Builds forward pass computation network.
      Returns:
        outputs: logits and loss of the model or logits.
      """
      build_network_result = self.model.build_network(
          input_list, is_training, n_classes)
      logits = build_network_result.logits
      if return_loss:
        loss = self.model.loss_function(input_list, build_network_result)
        return logits, loss
      return logits

    all_images, all_labels = [], []
    all_loss, all_loss_adv = [], []
    all_logits, all_logits_adv = [], []
    list_update_ops = []

    # Build the processing and model for the worker.
    with tf.name_scope("input"):
      input_list = self.reader.input_fn().get_next()

    for device_num in range(len(self.devices)):
      with tf.name_scope('tower_{}'.format(device_num)) as name_scope, (
          self.variable_mgr.create_outer_variable_scope(device_num)):
        inputs = input_list[device_num]
        with tf.device(self.devices[device_num]):
          logits, loss = forward_pass(inputs)
          loss = tf.reduce_mean(loss)
          all_images.append(inputs[0])
          all_labels.append(inputs[1])
          all_logits.append(logits)
          all_loss.append(loss)

          if self.params.eval_under_attack:
            with tf.device(self.devices[device_num]):
              adv = self.attack.generate(inputs[0],
                            partial(forward_pass, return_loss=False))
              input_adv = [adv, inputs[0]]
              logits_adv, loss_adv = forward_pass(input_adv)
              logits_adv.append(results['logits'])
              all_images_adv.append(adv)
              all_logits_adv.append(logits_adv)
              all_loss_adv.append(loss_adv)


    with tf.device(self.cpu_device):

      images = tf.concat(all_images, 0)
      labels = tf.concat(all_labels, 0)
      logits = tf.concat(all_logits, 0)
      loss = tf.reduce_mean(all_loss)

      predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
      loss_metric, loss_update_op = tf.metrics.mean(loss)
      accuracy, acc_update_op = tf.metrics.accuracy(
        tf.cast(labels, tf.float32), predictions)
      fetches['loss'] = loss_metric
      fetches['accuracy'] = accuracy
      list_update_ops.extend([loss_update_op, acc_update_op])

      if self.params.eval_under_attack:

        images_adv = tf.concat(all_images_adv, 0)
        logits_adv = tf.concat(all_logits_adv, 0)
        loss_adv = tf.reduce_mean(all_loss_adv)

        predictions_adv = tf.argmax(
          logits_adv, axis=1, output_type=tf.int32)
        loss_adv_metric, loss_adv_update_op = tf.metrics.mean(loss_adv)
        accuracy_adv, acc_adv_update_op = tf.metrics.accuracy(
          tf.cast(labels, tf.float32), predications_adv)
        fetches['loss_adv'] = loss_adv_metric
        fetches['accuracy_adv'] = accuracy_adv
        list_update_ops.extend([loss_adv_update_op, acc_adv_update_op])

        perturbation = images_adv - images
        perturbation = tf.layers.flatten(perturbation)
        for name, p in [('l1', 1), ('l2', 2), ('linf', np.inf)]:
          value, update = tf.metrics.mean(
            tf.norm(perturbation, ord=p, axis=1))
          fetches['mean_{}'.format(name)] = value
          list_update_ops.append(update)

    fetches['logits'] = all_logits
    fetches['loss'] = loss
    update_ops = tf.group(*list_update_ops)
    fetches['update_ops'] = update_ops
    return fetches

  def _run_fetches(self, sess, fetches):
     batch_start_time = time.time()
     results = sess.run(fetches)
     seconds_per_batch = time.time() - batch_start_time
     examples_per_second = self.batch_size / seconds_per_batch
     return results, examples_per_second


  def eval_attack(self, sess, fetches):
    """Run the evaluation under attack."""
    count = 0
    while True:
      try:
        results, examples_per_second = self._run_fetches(sess, fetches)
        count += self.batch_size
        if self.params.dump_files:
          dump.files(results)
        self.message.add('', [count, self.reader.n_test_files])
        self.message.add(
          'acc img/adv',
          [results['accuracy'], results['accuracy_adv']], format='.5f')
        self.message.add(
          'avg loss', [results['loss'], results['loss_adv']], format='.5f')
        self.message.add('imgs/sec', examples_per_second, format='.3f')
        norms_mean = [result['mean_l1'], result['mean_l2'], result['mean_linf']]
        self.message.add('l1/l2/linf mean', norms_mean, format='.2f')
        logging.info(self.message.get_message())
      except tf.errors.OutOfRangeError:
        self.message.add(
          'Final: images/adv',
          [results['accuracy'], results['accuracy_adv']], format='.5f')
        self.message.add(
          'avg loss', [results['loss'], results['loss_adv']], format='.5f')
        logging.info(self.message.get_message())
        logging.info("Done evaluation of adversarial examples.")
        return results['accuracy'], results['accuracy_adv']


  def eval_loop(self, sess, fetches, global_step):
    """Run the evaluation loop once."""
    while True:
      try:
        results, examples_per_second = self._run_fetches(sess, fetches)
        # if self.params.dump_files:
        #   self.dump.files(results)
        self.message.add('step', global_step)
        self.message.add('accuracy', results['accuracy'], format='.5f')
        self.message.add('avg loss', results['loss'], format='.5f')
        self.message.add('imgs/sec', examples_per_second, format='.0f')
        logging.info(self.message.get_message())
      except tf.errors.OutOfRangeError:
        if self.best_accuracy is None or self.best_accuracy < results['accuracy']:
          self.best_global_step = global_step
          self.best_accuracy = results['accuracy']
        if self.params.summary_verbosity > 0:
          make_summary("accuracy", results['accuracy'], self.summary_writer, global_step)
          make_summary("loss", results['loss'], self.summary_writer, global_step)
          self.summary_writer.flush()
        self.message.add('step', global_step)
        self.message.add('accuracy', results['accuracy'], format='.5f')
        self.message.add('avg loss', results['loss'], format='.5f')
        logging.info(self.message.get_message())
        logging.info("Done with batched inference.")
        return


