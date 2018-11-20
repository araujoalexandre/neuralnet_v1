
import os, sys
import json
import time

import readers
import models
import losses
from learning_rate import LearningRate
from optimizer import Optimizer
from gradients import ProcessGradients

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib

FLAGS = flags.FLAGS

# Dataset flags.
flags.DEFINE_string("train_dir", "/tmp/model/",
                    "The directory to save the model files in.")


flags.DEFINE_integer("batch_size", 32,
                     "How many examples to process per batch for training.")
flags.DEFINE_integer("num_epochs", 5,
                     "How many passes to make over the dataset before "
                     "halting training.")
flags.DEFINE_integer("max_steps", 1000000,
                     "How many steps to make before halting training.")
flags.DEFINE_bool("start_new_model", False,
                  "If set, this will not resume from a checkpoint and will "
                  "instead create a new model instance.")

# Model flags.
flags.DEFINE_string("model", "Model",
                    "Which architecture to use for the model. "
                    "Models are defined in models.py.")
flags.DEFINE_string("reader", "MNISTReader",
                    "Which reader to use for the data load."
                    "Readers are defined in readers.py")

# Training flags.
flags.DEFINE_integer("num_gpu", 1,
                     "The maximum number of GPU devices to use for training. "
                     "Flag only applies if GPUs are installed")
flags.DEFINE_string("label_loss", "SoftmaxCrossEntropyWithLogits",
                    "Which loss function to use for training the model.")
flags.DEFINE_float("regularization_penalty", 1.0,
                   "How much weight to give to the regularization loss "
                   "(the label loss has a weight of 1).")

# Other flags.
flags.DEFINE_bool("log_device_placement", False,
                  "Whether to write the device on which every op will run "
                  "into the logs on startup.")
flags.DEFINE_integer("save_checkpoint_steps", 1000,
  "The frequency, in number of global steps, that a checkpoint is saved "
  "using a default checkpoint saver. If both save_checkpoint_steps and "
  "save_checkpoint_secs are set to None, then the default checkpoint saver "
  "isn't used. If both are provided, then only save_checkpoint_secs is used. "
  "Default not enabled.")
flags.DEFINE_integer("save_summaries_steps", 120,
  "The frequency, in number of global steps, that the summaries are written "
  "to disk using a default summary saver. If both save_summaries_steps and "
  "save_summaries_secs are set to None, then the default summary saver isn't "
  "used.")
flags.DEFINE_integer("log_steps", 100,
                   "The frequency, in number of global steps, that the loss "
                   "is logged.")

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
  tower_predictions = []
  tower_label_losses = []
  for i in range(num_towers):
    # For some reason these 'with' statements can't be combined onto the same
    # line. They have to be nested.
    with tf.device(device_string.format(i)):
      with (tf.variable_scope("tower", reuse=True if i > 0 else None)):
        with (slim.arg_scope([slim.model_variable, slim.variable],
          device="/cpu:0" if num_gpus!=1 else "/gpu:0")):

          predictions = model.create_model(tower_inputs[i],
            labels=tower_labels[i], n_classes=10, is_training=True)
          tower_predictions.append(predictions)

          for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

          label_loss = label_loss_fn.calculate_loss(predictions, tower_labels[i])
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

  if regularization_penalty != 0:
    tf.summary.scalar("reg_loss", reg_loss)

  # process and apply gradients
  gradients = ProcessGradients(tower_gradients).get_gradients()
  train_op = opt.apply_gradients(gradients, global_step=global_step)

  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("learning_rate", learning_rate)
  tf.add_to_collection("predictions", tf.concat(tower_predictions, 0))
  tf.add_to_collection("images_batch", images_batch)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("train_op", train_op)


class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, model, reader,
                log_device_placement):
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
      log_device_placement=log_device_placement)
    self.model = model
    self.reader = reader

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model:
      self.remove_training_directory(self.train_dir)

    if not os.path.exists(self.train_dir):
      os.makedirs(self.train_dir)

    logging.info('{}: Parameters used:'.format(task_as_string(self.task)))
    for key, value in sorted(FLAGS.flag_values_dict().items()):
      if key not in ['h', 'help', 'helpfull', 'helpshort']:
        logging.info('{}: {}'.format(key, value))

    logging.info('Command used: ')
    logging.info('{} {}'.format('python3', ' '.join([x for x in sys.argv])))

    model_flags_dict = FLAGS.flag_values_dict()
    flags_json_path = os.path.join(FLAGS.train_dir, "model_flags.json")
    if os.path.exists(flags_json_path):
      existing_flags = json.load(open(flags_json_path))
      if existing_flags != model_flags_dict:
        logging.error("Model flags do not match existing file {}. Please "
                      "delete the file, change --train_dir, or pass flag "
                      "--start_new_model".format(flags_json_path))
        logging.error("Ran model with flags: {}".format(str(model_flags_dict)))
        logging.error("Previously ran with flags: {}".format(
          str(existing_flags)))
        exit(1)
    else:
      # Write the file.
      with open(flags_json_path, "w") as fout:
        fout.write(json.dumps(model_flags_dict))

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
        predictions = tf.get_collection("predictions")[0]
        labels = tf.get_collection("labels")[0]
        train_op = tf.get_collection("train_op")[0]
        init_op = tf.global_variables_initializer()

      hooks = [
        tf.train.NanTensorHook(loss),
        tf.train.StopAtStepHook(num_steps=FLAGS.max_steps)
      ]

      scaffold = tf.train.Scaffold(saver=saver, init_op=init_op)

      session_args = dict(
        is_chief=self.is_master,
        scaffold=scaffold,
        checkpoint_dir=FLAGS.train_dir,
        hooks=hooks,
        save_checkpoint_steps=FLAGS.save_checkpoint_steps,
        save_summaries_steps=FLAGS.save_summaries_steps,
        log_step_count_steps=10*FLAGS.log_steps,
        config=self.config,
      )

      batch_size = FLAGS.batch_size
      num_gpu = FLAGS.num_gpu

      logging.info("Start training")
      with tf.train.MonitoredTrainingSession(**session_args) as sess:
        while not sess.should_stop():
          try:
            batch_start_time = time.time()
            (_, global_step_val, loss_val, learning_rate_val,
              predictions_val, labels_val) = sess.run(
                [train_op, global_step, loss, learning_rate, predictions, labels])
            seconds_per_batch = time.time() - batch_start_time
            examples_per_second = labels_val.shape[0] / seconds_per_batch

            to_print = global_step_val % FLAGS.log_steps == 0
            if (self.is_master and to_print) or not global_step_val:
              epoch = ((global_step_val * batch_size * num_gpu)
                / self.reader.n_train_files)
              message = ("training epoch: {:.2f} | step: {} | lr: {:.6f} "
              "| loss: {:.2f} | Examples/sec: {:.0f}")
              logging.info(message.format(epoch,
                global_step_val, learning_rate_val,
                loss_val, examples_per_second))

          except tf.errors.OutOfRangeError:
            logging.info("{}: Done training -- epoch limit reached.".format(
              task_as_string(self.task)))
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

    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses, tf.nn])()

    build_graph(reader=reader,
                model=model,
                label_loss_fn=label_loss_fn,
                batch_size=FLAGS.batch_size * FLAGS.num_gpu,
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


def main(unused_argv):
  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("{}: Tensorflow version: {}.".format(
    task_as_string(task), tf.__version__))

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    reader = find_class_by_name(FLAGS.reader, [readers])(
      FLAGS.batch_size * FLAGS.num_gpu, num_epochs=FLAGS.num_epochs,
      is_training=True)

    model = find_class_by_name(FLAGS.model, [models])()
    logging.info("Using {} as model".format(FLAGS.model))
    
    trainer = Trainer(cluster, task, FLAGS.train_dir, model, reader,
      FLAGS.log_device_placement)
    trainer.run(start_new_model=FLAGS.start_new_model)

  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("{}: Invalid task_type: {}.".format(
      task_as_string(task), task.type))


if __name__ == "__main__":
  app.run()
