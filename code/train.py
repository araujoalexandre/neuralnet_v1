
import os, sys
import json
import time
import pprint
from os.path import join, exists
from datetime import datetime

import readers
import models
import losses
from learning_rate import LearningRate
from optimizer import Optimizer
from gradients import ProcessGradients

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib

from config import hparams as FLAGS


def task_as_string(task):
  return "/job:{}/task:{}".format(task.type, task.index)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def build_graph(reader, model, label_loss_fn, batch_size, regularization_penalty):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: the input class.
    model: The core model.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
  """
  global_step = tf.train.get_or_create_global_step()

  local_device_protos = device_lib.list_local_devices()
  gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
  gpus = gpus[:FLAGS.num_gpu]
  num_gpus = len(gpus)

  if num_gpus > 0:
    logging.info("Using the following GPUs to train: " + str(gpus))
    num_towers = num_gpus
    device_string = '/gpu:{}'
    logging.info("Using total batch size of {} for training "
      "over {} GPUs: batch size of {} per GPUs.".format(
        batch_size, num_towers, batch_size // num_towers))
  else:
    logging.info("No GPUs found. Training on CPU.")
    num_towers = 1
    device_string = '/cpu:{}'
    logging.info("Using total batch size of {} for training. ".format(
      batch_size))

  learning_rate = LearningRate(global_step, batch_size).get_learning_rate()
  opt = Optimizer(learning_rate).get_optimizer()

  with tf.name_scope("train_input"):
    images_batch, labels_batch = reader.input_fn()
  tf.summary.histogram("model/input_raw", images_batch)

  tower_inputs = tf.split(images_batch, num_towers)
  tower_labels = tf.split(labels_batch, num_towers)
  tower_gradients = []
  tower_logits = []
  tower_label_losses = []
  for i in range(num_towers):
    # For some reason these 'with' statements can't be combined onto the same
    # line. They have to be nested.
    with tf.device(device_string.format(i)):
      with (tf.variable_scope("tower", reuse=True if i > 0 else None)):
        with (slim.arg_scope([slim.model_variable, slim.variable],
          device="/cpu:0" if num_gpus!=1 else "/gpu:0")):

          logits = model.create_model(tower_inputs[i],
            labels=tower_labels[i], n_classes=10, is_training=True)
          tower_logits.append(logits)

          for variable in tf.trainable_variables():
            tf.summary.histogram(variable.op.name, variable)

          label_loss = label_loss_fn.calculate_loss(
            logits=logits, labels=tower_labels[i])
          reg_losses = tf.losses.get_regularization_losses()
          if reg_losses:
            reg_loss = tf.add_n(reg_losses)
          else:
            reg_loss = tf.constant(0.)

          # Adds update_ops (e.g., moving average updates in batch norm) as
          # a dependency to the train_op.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          if update_ops:
            with tf.control_dependencies(update_ops):
              barrier = tf.no_op(name="gradient_barrier")
              with tf.control_dependencies([barrier]):
                label_loss = tf.identity(label_loss)
          tower_label_losses.append(label_loss)

          # Incorporate the L2 weight penalties etc.
          final_loss = regularization_penalty * reg_loss + label_loss
          gradients = opt.compute_gradients(final_loss,
              colocate_gradients_with_ops=False)
          tower_gradients.append(gradients)

  label_loss = tf.reduce_mean(tf.stack(tower_label_losses))
  tf.summary.scalar("label_loss", label_loss)

  # process and apply gradients
  gradients = ProcessGradients(tower_gradients).get_gradients()
  train_op = opt.apply_gradients(gradients, global_step=global_step)

  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("learning_rate", learning_rate)
  tf.add_to_collection("summary_op", tf.summary.merge_all())
  tf.add_to_collection("train_op", train_op)


class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, model, reader, batch_size):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """
    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(allow_soft_placement=True,
      log_device_placement=FLAGS.log_device_placement)
    self.model = model
    self.reader = reader
    self.batch_size = batch_size

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model and exists(self.train_dir):
      self.remove_training_directory(self.train_dir)

    if not exists(self.train_dir):
      os.makedirs(self.train_dir)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(FLAGS.values())

    model_flags_dict = FLAGS.to_json()
    flags_json_path = join(self.train_dir, "model_flags.json")
    if exists(flags_json_path):
      existing_flags = json.load(open(flags_json_path))
      if existing_flags != model_flags_dict:
        logging.error("Model flags do not match existing file {}. Please "
                      "delete the file, change --train_dir, or pass flag "
                      "--start_new_model".format(flags_json_path))
        logging.error("Ran model with flags: {}".format(str(model_flags_dict)))
        logging.error("Previously ran with flags: {}".format(
          str(existing_flags)))
        sys.exit(1)
    else:
      # Write the file.
      with open(flags_json_path, "w") as fout:
        fout.write(model_flags_dict)

    target, device_fn = self.start_server_if_distributed()
    meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

    with tf.Graph().as_default() as graph:
      if meta_filename:
        saver = self.recover_model(meta_filename)

      with tf.device(device_fn):
        if not meta_filename:
          saver = self.build_model(self.model, self.reader)

        global_step = tf.train.get_global_step()
        loss = tf.get_collection("loss")[0]
        learning_rate = tf.get_collection("learning_rate")[0]
        train_op = tf.get_collection("train_op")[0]
        summary_op = tf.get_collection("summary_op")[0]
        init_op = tf.global_variables_initializer()

      hooks = [
        tf.train.NanTensorHook(loss),
        tf.train.StopAtStepHook(num_steps=FLAGS.max_steps)
      ]

      scaffold = tf.train.Scaffold(
        saver=saver,
        init_op=init_op,
        summary_op=summary_op,
      )

      session_args = dict(
        is_chief=self.is_master,
        scaffold=scaffold,
        checkpoint_dir=self.train_dir,
        hooks=hooks,
        save_checkpoint_steps=FLAGS.save_checkpoint_steps,
        save_summaries_steps=FLAGS.save_summaries_steps,
        log_step_count_steps=10*FLAGS.frequency_log_steps,
        config=self.config,
      )

      logging.info("Start training")
      with tf.train.MonitoredTrainingSession(**session_args) as sess:
        profiler = tf.profiler.Profiler(sess.graph)
        step = 0
        while not sess.should_stop():
          try:

            make_profile = False
            profile_args = {}

            if step % 1000 == 0 and FLAGS.profiler:
              make_profile = True
              run_meta = tf.RunMetadata()
              profile_args = {
                'options': tf.RunOptions(
                  trace_level=tf.RunOptions.FULL_TRACE),
                'run_metadata': run_meta
              }

            batch_start_time = time.time()
            (_, global_step_val, loss_val, learning_rate_val) = sess.run(
                [train_op, global_step, loss, learning_rate], **profile_args)
            seconds_per_batch = time.time() - batch_start_time
            examples_per_second = self.batch_size / seconds_per_batch

            if make_profile and FLAGS.profiler:
              profiler.add_step(step, run_meta)

              # Profile the parameters of your model.
              profiler.profile_name_scope(options=(tf.profiler.ProfileOptionBuilder
                  .trainable_variables_parameter()))

              # Or profile the timing of your model operations.
              opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
              profiler.profile_operations(options=opts)

              # Or you can generate a timeline:
              opts = (tf.profiler.ProfileOptionBuilder(
                      tf.profiler.ProfileOptionBuilder.time_and_memory())
                      .with_step(step)
                      .with_timeline_output('./profile.logs').build())
              profiler.profile_graph(options=opts)


            to_print = global_step_val % FLAGS.frequency_log_steps == 0
            if (self.is_master and to_print) or not global_step_val:
              epoch = ((global_step_val * self.batch_size)
                / self.reader.n_train_files)
              message = ("training epoch: {:.2f} | step: {} | lr: {:.6f} "
              "| loss: {:.2f} | Examples/sec: {:.0f}")
              logging.info(message.format(epoch,
                global_step_val, learning_rate_val,
                loss_val, examples_per_second))

            step += 1


          except tf.errors.OutOfRangeError:
            logging.info("{}: Done training -- epoch limit reached.".format(
              task_as_string(self.task)))
            profiler.advise()
            break
    logging.info("{}: Exited training loop.".format(task_as_string(self.task)))


  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("{}: Starting trainer within cluster {}.".format(
                   task_as_string(self.task), self.cluster.as_dict()))
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device=task_as_string(self.task),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info("{}: Removing existing train directory.".format(
        task_as_string(self.task)))
      gfile.DeleteRecursively(train_dir)
    except:
      logging.error("{}: Failed to delete directory {} when starting a new "
        "model. Please delete it manually and try again.".format(
          task_as_string(self.task), train_dir))
      sys.exit()


  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("{}: Flag 'start_new_model' is set. Building a new "
        "model.".format(task_as_string(self.task)))
      return None

    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint:
      logging.info("{}: No checkpoint file found. Building a new model.".format(
                   task_as_string(self.task)))
      return None

    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("{}: No meta graph file found. Building a new model.".format(
                     task_as_string(self.task)))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("{}: Restoring from meta graph file {}".format(
      task_as_string(self.task), meta_filename))
    return tf.train.import_meta_graph(meta_filename,
      clear_devices=FLAGS.clear_devices)

  def build_model(self, model, reader):
    """Find the model and build the graph."""

    label_loss_fn = find_class_by_name(FLAGS.loss, [losses, tf.nn])()

    build_graph(reader=reader,
                model=model,
                label_loss_fn=label_loss_fn,
                batch_size=self.batch_size,
                regularization_penalty=FLAGS.regularization_penalty)

    #TODO: make max_to_keep a FLAGS argument
    return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)


class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("{}: Starting parameter server within cluster {}.".format(
      task_as_string(self.task), self.cluster.as_dict()))
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """
  if not task.type:
    raise ValueError("{}: The task type must be specified.".format(
      task_as_string(task)))
  if task.index is None:
    raise ValueError("{}: The task index must be specified.".format(
      task_as_string(task)))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)


def main():

  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  if FLAGS.train_dir == "auto":
    dirname = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    train_dir = join(FLAGS.path, dirname)
  else:
    train_dir = join(FLAGS.path, FLAGS.train_dir)

  # Setup logging & log the version.
  logging.set_verbosity(logging.INFO)
  logging.info("{}: Tensorflow version: {}.".format(
    task_as_string(task), tf.__version__))

  if FLAGS.num_gpu == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
      map(str, range(FLAGS.num_gpu)))

  # Define batch size
  if FLAGS.num_gpu:
    batch_size = FLAGS.batch_size * FLAGS.num_gpu
  else:
    batch_size = FLAGS.batch_size

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    reader = find_class_by_name(FLAGS.reader, [readers])(
      batch_size, num_epochs=FLAGS.num_epochs,
      is_training=True)

    model = find_class_by_name(FLAGS.model, [models])()
    logging.info("Using {} as model".format(FLAGS.model))

    trainer = Trainer(cluster, task, train_dir, model, reader, batch_size)
    trainer.run(start_new_model=FLAGS.start_new_model)

  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("{}: Invalid task_type: {}.".format(
      task_as_string(task), task.type))


if __name__ == "__main__":
  main()
