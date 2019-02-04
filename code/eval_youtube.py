
import json
import time
import os
import sys
import re
import pprint
from os.path import join, basename, exists

import models
import losses
import readers
import eval_util
import utils

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
from tensorflow.python.lib.io import file_io

from config import YParams
from config import hparams as FLAGS

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


class Evaluate:

  def __init__(self):

   self.wait = 20

  def build_graph(self):
    """Creates the Tensorflow graph for evaluation.
    """
    global_step = tf.train.get_or_create_global_step()

    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpus = gpus[:FLAGS.eval_num_gpu]
    num_gpus = len(gpus)

    if num_gpus > 0:
      logging.info("Using the {} GPUs".format(num_gpus))
      num_towers = num_gpus
      device_string = '/gpu:{}'
      logging.info("Using total batch size of {} for evaluation "
        "over {} GPUs: batch size of {} per GPUs.".format(
          self.batch_size, num_towers, self.batch_size // num_towers))
    else:
      logging.info("No GPUs found. Eval on CPU.")
      num_towers = 1
      device_string = '/cpu:{}'
      logging.info("Using total batch size of {} for evalauton "
        "on CPU.".format(self.batch_size))

    with tf.name_scope("train_input"):
      images_batch, labels_batch = self.reader.input_fn()

    tower_inputs = tf.split(images_batch, num_towers)
    tower_labels = tf.split(labels_batch, num_towers)
    tower_logits, tower_label_losses = [], []
    for i in range(num_towers):
      # For some reason these 'with' statements can't be combined onto the same
      # line. They have to be nested.
      with tf.device(device_string.format(i)):
        with (tf.variable_scope("tower", reuse=True if i > 0 else None)):
          with (slim.arg_scope([slim.model_variable, slim.variable],
            device="/cpu:0" if num_gpus!=1 else "/gpu:0")):
            logits = self.model.create_model(tower_inputs[i],
              n_classes=self.reader.n_classes, is_training=False)
            tower_logits.append(logits)
            label_loss = self.loss_fn.calculate_loss(
              logits=logits, labels=tower_labels[i])
            tower_label_losses.append(label_loss)

    self.logits = tf.concat(tower_logits, 0)
    self.labels = tf.cast(labels_batch, tf.float32)
    self.labels_losses = tf.stack(tower_label_losses)
    self.summary_op = tf.summary.merge_all()

  def _get_global_step_from_ckpt(self, filename):
    regex = "(?<=ckpt-)[0-9]+"
    return int(re.findall(regex, filename)[-1])

  def get_checkpoint(self, last_global_step_val):
    if FLAGS.start_eval_from_ckpt:
      files = file_io.get_matching_files(
        join(self.train_dir, 'model.ckpt-*.index'))
      # No files
      if not files:
        return None, None
      files = sorted(files, key=self._get_global_step_from_ckpt)
      start_at = FLAGS.start_eval_from_ckpt
      if str(start_at).isdigit():
        start_at = int(start_at)
        files = list(filter(lambda x: self._get_global_step_from_ckpt(x) > start_at, files))
      for filename in files:
        filname_global_step = self._get_global_step_from_ckpt(filename)
        if last_global_step_val < filname_global_step:
          return filename[:-6], filname_global_step
      return None, None
    else:
      latest_checkpoint = tf.train.latest_checkpoint(self.train_dir)
      if latest_checkpoint is None:
        return None, None
      global_step = self._get_global_step_from_ckpt(latest_checkpoint)
      return latest_checkpoint, global_step


  def eval_loop(self, last_global_step_val, evl_metrics):
    """Run the evaluation loop once.

    Args:
      last_global_step_val: the global step used in the previous evaluation.

    Returns:
      The global_step used in the latest model.
    """
    latest_checkpoint, global_step_val = self.get_checkpoint(
      last_global_step_val)
    logging.info("latest_checkpoint: {}".format(latest_checkpoint))

    if latest_checkpoint is None or global_step_val == last_global_step_val:
      time.sleep(self.wait)
      return last_global_step_val

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
      logging.info("Loading checkpoint for eval: {}".format(latest_checkpoint))

      # Restores from checkpoint
      self.saver.restore(sess, latest_checkpoint)
      sess.run(tf.local_variables_initializer())

      evl_metrics.clear()

      train_gpu = FLAGS.train_num_gpu
      train_batch_size = FLAGS.train_batch_size
      n_train_files = self.reader.n_train_files
      if train_gpu:
        epoch = ((global_step_val*train_batch_size*train_gpu) / n_train_files)
      else:
        epoch = ((global_step_val*train_batch_size) / n_train_files)

      examples_processed = 0
      while True:
        try:
          batch_start_time = time.time()

          fetches = [self.logits, self.labels, self.labels_losses,
                     self.summary_op]
          logits_val, labels_val, loss_val, summary_val = sess.run(fetches)
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = self.batch_size / seconds_per_batch
          examples_processed += self.batch_size

          iteration_info_dict = evl_metrics.accumulate(logits_val, labels_val, loss_val)
          iteration_info_dict["examples_per_second"] = examples_per_second

          iterinfo = utils.AddGlobalStepSummary(
              self.summary_writer,
              global_step_val,
              iteration_info_dict,
              summary_scope="Eval")
          logging.info("examples_processed: %d | %s", examples_processed,
                       iterinfo)

        except tf.errors.OutOfRangeError as e:
          logging.info(
              "Done with batched inference. Now calculating global performance "
              "metrics.")
          # calculate the metrics for the entire epoch
          epoch_info_dict = evl_metrics.get()
          epoch_info_dict["epoch_id"] = global_step_val

          self.summary_writer.add_summary(summary_val, global_step_val)
          epochinfo = utils.AddEpochSummary(
              self.summary_writer,
              global_step_val,
              epoch_info_dict,
              summary_scope="Eval")
          logging.info(epochinfo)
          evl_metrics.clear()

          if FLAGS.stopped_at_n:
           self.counter += 1
          break

        except Exception as e:
          logging.info("Unexpected exception: {}".format(e))
          sys.exit(0)

      return global_step_val

  def load_last_train_dir(self):
    while True:
      folders = tf.gfile.Glob(join(FLAGS.path, "*"))
      folders = list(filter(lambda x: "logs" not in x, folders))
      folders = sorted(folders, key=lambda x: basename(x))
      if folders:
        break
    return folders[-1]

  def load_config(self, train_dir):
    # Write json of flags
    model_flags_path = join("{}_logs".format(train_dir), "model_flags.yaml")
    if not exists(model_flags_path):
      raise IOError("Cannot find file {}. Did you run train.py on the same "
                    "--train_dir?".format(model_flags_path))
    flags_dict = YParams(model_flags_path, "eval")
    return flags_dict

  def run(self):

    tf.set_random_seed(0)  # for reproducibility

    # Setup logging & log the version.
    tf.set_random_seed(0)  # for reproducibility

    # Setup logging & log the version.
    tf.logging.set_verbosity(logging.INFO)
    logging.info("Tensorflow version: {}.".format(tf.__version__))

    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
      if FLAGS.eval_num_gpu == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
      else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
          map(str, range(FLAGS.eval_num_gpu)))

    # self.train_dir = join(FLAGS.path, FLAGS.train_dir)
    self.train_dir = FLAGS.train_dir

    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(FLAGS.values()))

    with tf.Graph().as_default():
      if FLAGS.eval_num_gpu:
        self.batch_size = \
            FLAGS.eval_batch_size * FLAGS.eval_num_gpu
      else:
        self.batch_size = FLAGS.eval_batch_size

      self.reader = find_class_by_name(FLAGS.reader, [readers])(
        self.batch_size, is_training=False)
      self.model = find_class_by_name(FLAGS.model, [models])()
      self.loss_fn = find_class_by_name(FLAGS.loss, [losses])()

      data_pattern = FLAGS.data_pattern
      if data_pattern is "":
        raise IOError("'data_pattern' was not specified. "
          "Nothing to evaluate.")

      self.build_graph()
      logging.info("Built evaluation graph")

      self.saver = tf.train.Saver(tf.global_variables())
      filename_suffix= "_{}_{}".format("eval",
                  re.findall("[a-z0-9]+", data_pattern.lower())[0])
      self.summary_writer = tf.summary.FileWriter(
        self.train_dir,
        filename_suffix=filename_suffix,
        graph=tf.get_default_graph())

      evl_metrics = eval_util.EvaluationMetrics(self.reader.n_classes, 20)

      self.counter = 0
      last_global_step_val = 0
      while self.counter < FLAGS.stopped_at_n:
        last_global_step_val = self.eval_loop(last_global_step_val, evl_metrics)
      logging.info("Done evaluation -- number of eval reached.")



if __name__ == '__main__':
  evaluate = Evaluate()
  evaluate.run()
