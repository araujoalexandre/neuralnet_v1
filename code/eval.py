
import json
import time
import os
import re
import pprint
from os.path import join, basename, exists

import models
import losses
import readers

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

  def make_summary(self, name, value, global_step_val):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    self.summary_writer.add_summary(summary, global_step_val)

  def build_graph(self):
    """Creates the Tensorflow graph for evaluation.
    """
    global_step = tf.train.get_or_create_global_step()

    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpus = gpus[:self.flags_dict.num_gpu]
    num_gpus = len(gpus)

    if num_gpus > 0:
      logging.info("Using the following GPUs to train: " + str(gpus))
      num_towers = num_gpus
      device_string = '/gpu:{}'
    else:
      logging.info("No GPUs found. Training on CPU.")
      num_towers = 1
      device_string = '/cpu:{}'

    logging.info("Using total batch size of {} for training "
      "over {} GPUs: batch size of {} per GPUs.".format(
        self.batch_size, num_towers, self.batch_size // num_towers))
    with tf.name_scope("train_input"):
      images_batch, labels_batch = self.reader.input_fn()
    tf.summary.histogram("model/input_raw", images_batch)

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
              labels=tower_labels[i], n_classes=self.reader.n_classes, is_training=False)
            tower_logits.append(logits)
            label_loss = self.loss_fn.calculate_loss(
              logits=logits, labels=tower_labels[i])
            tower_label_losses.append(label_loss)

    logits = tf.concat(tower_logits, 0)
    self.predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    self.labels = tf.cast(labels_batch, tf.float32)
    self.summary_op = tf.summary.merge_all()

    self.loss, self.loss_update_op = tf.metrics.mean(tf.stack(
                                                     tower_label_losses))
    self.accuracy, self.acc_update_op = tf.metrics.accuracy(self.labels,
                                                            self.predictions)

  def _get_global_step_from_ckpt(self, filename):
    regex = "(?<=ckpt-)[0-9]+"
    return int(re.findall(regex, filename)[-1])

  def get_checkpoint(self, last_global_step_val):
    if self.flags_dict.start_eval_from_ckpt == 'first':
      files = file_io.get_matching_files(
        join(self.train_dir, 'model.ckpt-*.index'))
      # No files
      if not files:
        return None
      sort_fn = lambda x: int(re.findall("(?<=ckpt-)[0-9]+", x)[-1])
      files = sorted(files, key=self._get_global_step_from_ckpt)
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


  def eval_loop(self, last_global_step_val):
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

    with tf.Session() as sess:
      logging.info("Loading checkpoint for eval: {}".format(latest_checkpoint))

      # Restores from checkpoint
      self.saver.restore(sess, latest_checkpoint)
      sess.run(tf.local_variables_initializer())
      while True:
        try:
          batch_start_time = time.time()
          sess.run([self.loss_update_op, self.acc_update_op, self.summary_op])
          loss_val, accuracy_val = sess.run([self.loss, self.accuracy])
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = self.batch_size / seconds_per_batch

          msg = ("global_step {} | Acc Accuracy: {:.5f} | Avg Loss: {:.5f} "
            "| Examples_per_sec: {:.2f}")
          logging.info(msg.format(global_step_val, accuracy_val, loss_val,
            examples_per_second))

        except tf.errors.OutOfRangeError:
          logging.info("Done with batched inference.")

          self.make_summary("Epoch/Accuracy", accuracy_val, global_step_val)
          self.make_summary("Epoch/Loss", loss_val, global_step_val)
          self.summary_writer.flush()

          msg = ("epoch/eval number {} | Accuracy: {:.5f} | Avg_Loss: {:5f}")
          logging.info(msg.format(global_step_val, accuracy_val, loss_val))
          break

        except Exception as e:
          logging.info("Unexpected exception: {}".format(e))
          break

      return global_step_val


  def load_last_train_dir(self):
    folders = tf.gfile.Glob(join(self.flags_dict.path, "*"))
    folders = sorted(folders, key=lambda x: basename(x))
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
    tf.logging.set_verbosity(logging.INFO)
    logging.info("Tensorflow version: {}.".format(tf.__version__))

    if FLAGS.num_gpu == 0:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        map(str, range(FLAGS.num_gpu)))

    if FLAGS.train_dir == "auto":
      self.flags_dict = FLAGS.values()
      self.train_dir = self.load_last_train_dir()
    else:
      self.train_dir = join(FLAGS.path, FLAGS.train_dir)
      self.flags_dict = self.load_config(self.train_dir)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(self.flags_dict)

    with tf.Graph().as_default():

      if self.flags_dict.num_gpu:
        self.batch_size = self.flags_dict.batch_size * self.flags_dict.num_gpu
      else:
        self.batch_size = self.flags_dict.batch_size

      self.reader = find_class_by_name(self.flags_dict.reader, [readers])(
        self.batch_size, is_training=False)
      self.model = find_class_by_name(self.flags_dict.model, [models])()
      self.loss_fn = find_class_by_name(self.flags_dict.loss, [losses])()

      if self.flags_dict.data["data_pattern"] is "":
        raise IOError("'data_pattern' was not specified. "
          "Nothing to evaluate.")

      self.build_graph()
      logging.info("Built evaluation graph")

      self.saver = tf.train.Saver(tf.global_variables())
      self.summary_writer = tf.summary.FileWriter(self.train_dir,
        graph=tf.get_default_graph())

      last_global_step_val = 0
      while True:
        last_global_step_val = self.eval_loop(last_global_step_val)
        # if FLAGS.run_once or FLAGS.checkpoint is not None:
        #   break



if __name__ == '__main__':
  evaluate = Evaluate()
  evaluate.run()
