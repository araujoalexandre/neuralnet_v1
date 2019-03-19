
import json
import time
import os
import re
import pprint
import pickle
from collections import OrderedDict
from functools import partial
from os.path import join, basename, exists, normpath

import models
from train_utils import losses
from dataset import readers

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
from tensorflow.python.lib.io import file_io

import attacks

from config import YParams
from config import hparams as FLAGS


def pickle_dump(file, path):
    """
        function to dump picke object
    """
    with open(path, 'wb') as f:
        pickle.dump(file, f, -1)

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


class Evaluate:

  def __init__(self):
   self.wait = 20

  def _predict(self, images_batch, num_towers, device_string, n_classes,
               is_training):

    tower_inputs = tf.split(images_batch, self.num_towers)
    tower_logits = []
    for i in range(num_towers):
      with tf.device(device_string.format(i)):
        reuse = bool(i > 0)
        with tf.variable_scope("tower", reuse=tf.AUTO_REUSE):
          logits = self.model.create_model(
            tower_inputs[i], n_classes, is_training)
          tower_logits.append(logits)
    logits_batch = tf.concat(tower_logits, 0)
    return logits_batch

  def build_graph(self):
    """Creates the Tensorflow graph for evaluation.
    """
    global_step = tf.train.get_or_create_global_step()

    with tf.name_scope("train_input"):
      images_batch, self.labels_batch = self.reader.input_fn()
      tf.summary.histogram("model/input_raw", images_batch)
      self.images_batch = images_batch

    # get loss and logits from real examples
    logits_batch = self._predict(
      images_batch, self.num_towers, self.device_string,
      self.reader.n_classes, False)

    losses_batch = self.loss_fn.calculate_loss(
      logits=logits_batch, labels=self.labels_batch)
    preds_batch = tf.argmax(logits_batch, axis=1, output_type=tf.int32)

    # get loss and logits from adv examples
    def fn_logits(x):
      return self._predict(x, self.num_towers,
                self.device_string, self.reader.n_classes, False)

    self.images_adv_batch = self.attack.generate(images_batch, fn_logits)
    logits_adv_batch = self._predict(
      self.images_adv_batch, self.num_towers, self.device_string,
      self.reader.n_classes, False)

    losses_adv_batch = self.loss_fn.calculate_loss(
      logits=logits_adv_batch, labels=self.labels_batch)
    preds_adv_batch = tf.argmax(
      logits_adv_batch, axis=1, output_type=tf.int32)

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
      log_device_placement=False,
      allow_soft_placement=True
    )

    # generate adv exemples
    with tf.Session(config=config).as_default()as sess:
      logging.info("Evaluation under attack:")

      self.attack.sess = sess

      # Restores from checkpoint
      self.saver.restore(sess, best_checkpoint)
      sess.run(tf.local_variables_initializer())

      fetches = OrderedDict(
         loss_update_op=self.loss_update_op,
         acc_update_op=self.acc_update_op,
         loss_adv_update_op=self.loss_adv_update_op,
         acc_adv_update_op=self.acc_adv_update_op,
         images=self.images_batch,
         images_adv=self.images_adv_batch,
         loss=self.loss,
         acc=self.acc,
         preds=self.preds,
         preds_adv=self.preds_adv,
         loss_adv=self.loss_adv,
         acc_adv=self.acc_adv,
         labels_batch=self.labels_batch,
      )

      count = 0
      id = 0
      while True:
        try:

          batch_start_time = time.time()
          values = sess.run(list(fetches.values()))
          fetches_values = OrderedDict(zip(fetches.keys(), values))
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = self.batch_size / seconds_per_batch

          images_val = fetches_values['images']
          images_adv_val = fetches_values['images_adv']
          loss_val = fetches_values['loss']
          acc_val = fetches_values['acc']
          preds_val = fetches_values['preds']
          preds_adv_val = fetches_values['preds_adv']
          loss_adv_val = fetches_values['loss_adv']
          acc_adv_val = fetches_values['acc_adv']
          labels = fetches_values['labels_batch']
          count += self.batch_size

          # dump images and images_adv
          sample = FLAGS.attack_sample
          name_var = dict(train_dir=train_dir, sample=sample, id=id)
          path_img_adv = '{train_dir}_logs/dump_carlini_images_adv_{sample}_{id}.pkl'
          path_img     = '{train_dir}_logs/dump_carlini_images_{sample}_{id}.pkl'
          path_img_adv = path_img_adv.format(**name_var)
          path_img     = path_img.format(**name_var)
          pickle_dump(images_adv_val, path_img_adv)
          pickle_dump(images_val, path_img)
          id += 1

          message_data = {
            'count': count,
            'total': self.reader.n_test_files,
            'acc': acc_val,
            'acc_adv': acc_adv_val,
            'loss': loss_val,
            'loss_adv': loss_adv_val,
            'imgs_sec': examples_per_second,
          }
          message = ("{count}/{total} | "
                     "images/adv: Acc : {acc:.5f}/{acc_adv:.5f} | "
                     "avg loss: {loss:.5f}/{loss_adv:.5f} | "
                     "imgs/sec: {imgs_sec:.2f}")
          logging.info(message.format(**message_data))

        except tf.errors.OutOfRangeError:

          msg = ("Final: images/adv: Acc: {:.5f}/{:.5f} | Avg Loss: {:.5f}/{:.5f}")
          logging.info(msg.format(acc_val, acc_adv_val, loss_val, loss_adv_val))
          logging.info("Done evaluation of adversarial examples.")
          break
    return acc_val, acc_adv_val


  def run(self):

    self.flags_dict = FLAGS

    # Setup logging & log the version.
    tf.logging.set_verbosity(logging.INFO)
    logging.info("Tensorflow version: {}.".format(tf.__version__))

    if FLAGS.eval_num_gpu == 0:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        map(str, range(FLAGS.eval_num_gpu)))

    train_dir = join(FLAGS.path, FLAGS.train_dir)

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

    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(self.flags_dict.values()))

    with tf.Graph().as_default():

      attack_method = FLAGS.attack_method
      attack_cls = getattr(attacks, attack_method, None)
      if attack_cls is None:
        raise ValueError("Attack is not recognized.")
      attack_config = getattr(FLAGS, attack_method)
      self.attack = attack_cls(
        batch_size=self.batch_size, sample=FLAGS.attack_sample, **attack_config)

      self.reader = find_class_by_name(self.flags_dict.reader, [readers])(
        self.batch_size, is_training=False)

      self.model = find_class_by_name(self.flags_dict.model, [models])()
      self.loss_fn = find_class_by_name(self.flags_dict.loss, [losses])()

      data_pattern = self.flags_dict.data_pattern
      self.dataset = re.findall("[a-z0-9]+", data_pattern.lower())[0]
      if data_pattern is "":
        raise IOError("'data_pattern' was not specified. "
          "Nothing to evaluate.")

      self.build_graph()
      self.saver = tf.train.Saver(tf.global_variables(scope="tower"))
      logging.info("Built evaluation graph")

      acc_val, acc_adv_val = self.eval(train_dir)

      record_file_name = "score_{}.txt".format(self.attack.get_name())
      record_file = join("{}_logs".format(train_dir), record_file_name)
      with open(record_file, 'w') as f:
        f.write("{:.5f}\t{:.5f}\n".format(acc_val, acc_adv_val))


if __name__ == '__main__':
  evaluate = Evaluate()
  evaluate.run()
