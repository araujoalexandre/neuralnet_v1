
import json
import time
import os
import re
import pickle
import pprint
from os.path import join, basename, exists, normpath

import models
import losses
import readers

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
from tensorflow.python.lib.io import file_io

from attacks import Attacks
from cleverhans.model import Model

from config import YParams
from config import hparams as FLAGS


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


# We have to wrap our model to match the cleverhans API
class ModelWrapper(Model):

  def __init__(self, model, n_classes=None, is_training=False,
               num_towers=None, device_string=None, batch_shape=None):

    Model.__init__(self, 'model', n_classes, locals())
    self.model = model
    self.n_classes = n_classes
    self.is_training = is_training
    self.num_towers = num_towers
    self.device_string = device_string
    self.batch_shape = batch_shape

    # Do a dummy run of fprop to make sure the variables 
    # are created from the start
    # self.fprop(tf.placeholder(tf.float32, [*self.batch_shape]))
    # Put a reference to the params in self so that the params get pickled
    # self.params = self.get_params()

  def fprop(self, x, **kwargs):

    tower_inputs = tf.split(x, self.num_towers)
    tower_logits = []
    for i in range(self.num_towers):
      # For some reason these 'with' statements can't be combined onto the same
      # line. They have to be nested.
      with tf.device(self.device_string.format(i)):
        reuse = bool(i > 0)
        with (tf.variable_scope("tower", reuse=tf.AUTO_REUSE)):
          with (slim.arg_scope([slim.model_variable, slim.variable],
                               device=self.device_string.format(0))):
            logits = self.model.create_model(tower_inputs[i], self.n_classes,
                                is_training=self.is_training)
            tower_logits.append(logits)

    logits = tf.concat(tower_logits, 0)
    ret = {
      'logits': logits,
      'probs': tf.nn.softmax(logits=logits)
    }
    return ret


class Evaluate:

  def __init__(self):
   self.wait = 20

  def _predict(self, images_batch, labels_batch, attack=False):

    logits_batch = self.model.fprop(images_batch, test="predict")['logits']
    loss_fn = self.loss_fn.calculate_loss
    losses_batch = loss_fn(logits=logits_batch, labels=labels_batch)

    preds_batch = tf.argmax(logits_batch, axis=1, output_type=tf.int32)
    return logits_batch, preds_batch, losses_batch

  def build_graph(self):
    """Creates the Tensorflow graph for evaluation.
    """
    global_step = tf.train.get_or_create_global_step()

    with tf.name_scope("train_input"):
      batch_shape = self.reader.batch_shape
      self.images_batch = tf.placeholder(tf.float32, batch_shape)
      self.images_adv_batch = tf.placeholder(tf.float32, batch_shape)
      self.labels_batch = tf.placeholder(tf.int32, (None, 1))
      tf.summary.histogram("model/input_raw", self.images_batch)
      tf.summary.histogram("model/input_raw_adv", self.images_adv_batch)

    # get loss and logits from real examples
    (logits_batch, preds_batch, losses_batch) = self._predict(
       self.images_batch, self.labels_batch, attack=False)

    (logits_adv_batch, preds_adv_batch, losses_adv_batch) = self._predict(
      self.images_adv_batch, self.labels_batch, attack=True)

    self.loss, self.loss_update_op = tf.metrics.mean(losses_batch)
    self.acc, self.acc_update_op = tf.metrics.accuracy(
       self.labels_batch, preds_batch)

    self.preds = tf.nn.softmax(logits_batch)
    self.preds_adv = tf.nn.softmax(logits_adv_batch)

    self.loss_adv, self.loss_adv_update_op = tf.metrics.mean(losses_adv_batch)
    self.acc_adv, self.acc_adv_update_op = tf.metrics.accuracy(
        self.labels_batch, preds_adv_batch)

  def get_best_checkpoint(self, train_dir):
    best_acc_file = join("{}_logs".format(train_dir),"best_accuracy.txt")
    with open(best_acc_file) as f:
      content = f.readline().split('\t')
      best_ckpt = content[0]
    best_ckpt_path = file_io.get_matching_files(
        join(train_dir, 'model.ckpt-{}.index'.format(best_ckpt)))
    return best_ckpt_path[-1][:-6], int(best_ckpt)

  def eval(self, train_dir):
    """Run the evaluation under attack.

    Args:
      last_global_step_val: the global step used in the previous evaluation.

    Returns:
      The global_step used in the latest model.
    """
    best_checkpoint, global_step_val = self.get_best_checkpoint(train_dir)

    train_gpu = self.flags_dict.train_num_gpu
    train_batch_size = self.flags_dict.train_batch_size
    n_train_files = self.reader.n_train_files
    if train_gpu:
      epoch = ((global_step_val*train_batch_size*train_gpu) / n_train_files)
    else:
      epoch = ((global_step_val*train_batch_size) / n_train_files)

    config = tf.ConfigProto(
      # device_count={'GPU': 2},
      log_device_placement=False,
    )

    # generate adv exemples
    with tf.Session(config=config) as sess:
      logging.info("Generate adv examples.")

      # Restores from checkpoint
      self.saver.restore(sess, best_checkpoint)
      sess.run(tf.local_variables_initializer())
      # init attack method
      attack_cls = Attacks(sess, self.model)
      logging.info("Init attack method")

      # get tf variables from graph
      fetches = [
        self.loss_update_op, self.acc_update_op,
        self.loss_adv_update_op, self.acc_adv_update_op,
        self.loss, self.acc, self.preds, self.preds_adv,
        self.loss_adv, self.acc_adv]

      model_proba, model_proba_adv = [], []
      while True:
        try:
          batch_start_time = time.time()
          # fetch images and labels from tf.data to numpy array
          # not very efficient but otherwise difficult with cleverhans
          imgs, labels = sess.run([self.reader_images, self.reader_labels])

          # generate adversarial images from cleverhans
          imgs_adv = attack_cls.generate(imgs, labels)

          # feed the images and adversarial images 
          # into the graph to make get the prediction
          feed_data_dict = {
            self.images_batch: imgs,
            self.images_adv_batch: imgs_adv,
            self.labels_batch: labels
          }
          (*_, loss_val, acc_val, preds_val,
           preds_adv_val, loss_adv_val, acc_adv_val) = \
              sess.run(fetches, feed_dict=feed_data_dict)
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = self.batch_size / seconds_per_batch

          msg = ("images/adv: Acc : {:.5f}/{:.5f}"
                   " | Avg Loss: {:.5f}/{:.5f} | imgs/sec: {:.2f}")
          logging.info(msg.format(acc_val, acc_adv_val, loss_val,
                      loss_adv_val, examples_per_second))

        except tf.errors.OutOfRangeError:
          msg = ("final: images/adv: Acc: {:.5f}/{:.5f} | Avg Loss: {:.5f}/{:.5f}")
          logging.info(msg.format(acc_val, acc_adv_val, loss_val, loss_adv_val))
          logging.info("Done evaluation of adversarial examples.")
          break
    return acc_val, acc_adv_val


  def run(self):

    tf.set_random_seed(0)  # for reproducibility
    self.flags_dict = FLAGS

    # Setup logging & log the version.
    tf.logging.set_verbosity(logging.INFO)
    logging.info("Tensorflow version: {}.".format(tf.__version__))

    if FLAGS.eval_num_gpu == 0:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        map(str, range(FLAGS.eval_num_gpu)))

    if self.flags_dict.eval_num_gpu:
      self.batch_size = \
          self.flags_dict.eval_batch_size * self.flags_dict.eval_num_gpu
    else:
      self.batch_size = self.flags_dict.eval_batch_size

    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpus = gpus[:self.flags_dict.eval_num_gpu]
    num_gpus = len(gpus)

    if num_gpus > 0:
      logging.info("Using the {} GPUs".format(num_gpus))
      self.num_towers = num_gpus
      self.device_string = '/gpu:{}'
      logging.info("Using total batch size of {} for evaluation "
        "over {} GPUs: batch size of {} per GPUs.".format(
          self.batch_size, self.num_towers,
            self.batch_size // self.num_towers))
    else:
      logging.info("No GPUs found. Eval on CPU.")
      self.num_towers = 1
      self.device_string = '/cpu:{}'
      logging.info("Using total batch size of {} for evalauton "
        "on CPU.".format(self.batch_size))

    train_dir = join(FLAGS.path, FLAGS.train_dir)

    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(self.flags_dict.values()))

    with tf.Graph().as_default():

      self.reader = find_class_by_name(self.flags_dict.reader, [readers])(
        self.batch_size, is_training=False)
      self.reader_images, self.reader_labels = self.reader.input_fn()

      model = find_class_by_name(self.flags_dict.model, [models])()
      self.model = ModelWrapper(model,
                    n_classes=self.reader.n_classes,
                    is_training=False,
                    num_towers=self.num_towers,
                    device_string=self.device_string,
                    batch_shape=self.reader.batch_shape)
      self.loss_fn = find_class_by_name(self.flags_dict.loss, [losses])()

      data_pattern = self.flags_dict.data_pattern
      self.dataset = re.findall("[a-z0-9]+", data_pattern.lower())[0]
      if data_pattern is "":
        raise IOError("'data_pattern' was not specified. "
          "Nothing to evaluate.")

      self.build_graph()
      self.saver = tf.train.Saver(tf.global_variables())
      logging.info("Built evaluation graph")

      acc_val, acc_adv_val = self.eval(train_dir)

      attack = self.flags_dict.attack_method
      record_file_name = "score_{}.txt".format(attack)
      record_file = join("{}_logs".format(train_dir),
                         record_file_name)
      with open(record_file, 'w') as f:
        f.write("{:.5f}\t{:.5f}\n".format(acc_val, acc_adv_val))


if __name__ == '__main__':
  evaluate = Evaluate()
  evaluate.run()
