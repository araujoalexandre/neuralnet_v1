
import os
import re
import time
import pprint
from os.path import exists, join, basename

import models
import losses
import readers

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import logging
from tensorflow.python.client import device_lib
from tensorflow.python.lib.io import file_io

from config import YParams
from config import hparams as FLAGS


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


class Attack:

  def _load_last_train_dir(self):
    folders = tf.gfile.Glob(join(FLASG.path, "*"))
    folders = sorted(folders, key=lambda x: basename(x))
    return folders[-1]

  def _get_attack_fgsm(self, images, logits):
    eps = self.attack_config["epsilon"]
    loss_fn = self.loss_fn.calculate_loss
    target = tf.argmax(logits, axis=1)
    target = tf.reshape(target, (-1, 1))
    loss = loss_fn(labels=target, logits=logits)
    dy_dx, = tf.gradients(loss, images)
    xadv = images + eps * tf.sign(dy_dx)
    xadv = tf.clip_by_value(xadv, -1.0, 1.0)
    return tf.stop_gradient(xadv)

  def _predict(self, images_batch, labels_batch, attack=False):

    n_classes = self.reader.n_classes
    loss_fn = self.loss_fn.calculate_loss

    tower_inputs = tf.split(images_batch, self.num_towers)
    tower_labels = tf.split(labels_batch, self.num_towers)
    tower_logits, tower_label_losses = [], []
    for i in range(self.num_towers):
      # For some reason these 'with' statements can't be combined onto the same
      # line. They have to be nested.
      with tf.device(self.device_string.format(i)):
        reuse = True if i > 0 or attack else None
        with (tf.variable_scope("tower", reuse=reuse)):
          with (slim.arg_scope([slim.model_variable, slim.variable],
            device="/cpu:0" if self.num_towers !=1 else "/gpu:0")):
            logits = self.model.create_model(tower_inputs[i],
              labels=tower_labels[i], n_classes=n_classes, is_training=False)
            label_loss = loss_fn(logits=logits, labels=tower_labels[i])
            tower_logits.append(logits)
            tower_label_losses.append(label_loss)

    logits_batch = tf.concat(tower_logits, 0)
    preds_batch = tf.argmax(logits, axis=1, output_type=tf.int32)
    losses_batch = tf.stack(tower_label_losses)

    return logits_batch, preds_batch, losses_batch

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
      self.num_towers = num_gpus
      self.device_string = '/gpu:{}'
    else:
      logging.info("No GPUs found. Training on CPU.")
      self.num_towers = 1
      self.device_string = '/cpu:{}'

    logging.info("Using total batch size of {} for training "
      "over {} GPUs: batch size of {} per GPUs.".format(
        self.batch_size, self.num_towers, self.batch_size // self.num_towers))
    with tf.name_scope("train_input"):
      images_batch, labels_batch = self.reader.input_fn()
    tf.summary.histogram("model/input_raw", images_batch)

    # get loss and logits from real examples
    (logits_batch, preds_batch, losses_batch) = self._predict(
        images_batch, labels_batch, attack=False)

    # get loss and logits from adversarial examples
    adversarial_images_batch = self._get_attack(images_batch, logits_batch)
    (attack_logits_batch, attack_preds_batch,
      attack_losses_batch) = self._predict(
        adversarial_images_batch, labels_batch, attack=True)

    self.loss, self.loss_update_op = tf.metrics.mean(losses_batch)
    self.accuracy, self.acc_update_op = tf.metrics.accuracy(
       labels_batch, preds_batch)

    self.loss_attack, self.loss_attack_update_op = tf.metrics.mean(
        attack_losses_batch)
    self.accuracy_attack, self.acc_attack_update_op = tf.metrics.accuracy(
        labels_batch, attack_preds_batch)


  def _eval_with_attack(self, checkpoint_path):
    """Run the evaluation with attack.
    """
    with tf.Session() as sess:

      loss, acc = self.loss, self.accuracy
      loss_update_op, acc_update_op = self.loss_update_op, self.acc_update_op

      loss_attack, acc_attack = self.loss_attack, self.accuracy_attack
      loss_attack_update_op = self.loss_attack_update_op
      acc_attack_update_op = self.acc_attack_update_op

      # Restores from checkpoint
      self.saver.restore(sess, checkpoint_path)
      sess.run(tf.local_variables_initializer())
      while True:
        try:
          batch_start_time = time.time()

          run = [loss, acc,
                 loss_update_op, acc_update_op,
                 loss_attack, acc_attack,
                 loss_attack_update_op, acc_attack_update_op]

          (loss_val, acc_val, _, _,
           loss_attack_val, acc_attack_val, _, _) = sess.run(run)

          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = self.batch_size / seconds_per_batch

          msg = ("No Attack/Attack : Acc : {:.5f}/{:.5f}"
                 " | Avg Loss: {:.5f}/{:.5f} | Examples_per_sec: {:.2f}")
          logging.info(msg.format(acc_val, acc_attack_val, loss_val,
                    loss_attack_val, examples_per_second))

        except tf.errors.OutOfRangeError:
          logging.info("Done with batched inference.")

          msg = ("No Attack/Attack: Acc: {:.5f}/{:.5f}"
                 " | Avg Loss: {:.5f}/{:.5f}")
          logging.info(msg.format(acc_val, acc_attack_val,
                                  loss_val, loss_attack_val))
          break

        except Exception as e:
          logging.info("Unexpected exception: {}".format(e))
          break

  def load_last_train_dir(self):
    folders = tf.gfile.Glob(join(self.flags_dict.path, "*"))
    folders = list(filter(lambda x: "logs" not in x, folders))
    folders = sorted(folders, key=lambda x: basename(x))
    last_train_dir = folders[-1]
    return last_train_dir

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
      self.flags_dict = FLAGS
      self.train_dir = self.load_last_train_dir()
    else:
      self.train_dir = join(FLAGS.path, FLAGS.train_dir)
      self.flags_dict = self.load_config(self.train_dir)

    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(self.flags_dict.values()))

    self.attack_config = FLAGS.attack
    self.method = self.attack_config["method"]
    self._get_attack = getattr(self,
            "_get_attack_{}".format(self.method.lower()))
    self.checkpoint = FLAGS.attack["checkpoint"]


    if self.checkpoint == "auto":
      checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
    else:
      checkpoint_path = join(self.train_dir, self.checkpoint)
      if not exists(checkpoint_path):
        logging.error("checkpoint {} not found in {}.".format(
            self.checkpoint, self.train_dir))

    logging.info("Loading checkpoint for eval: {}".format(checkpoint_path))


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
      self._eval_with_attack(checkpoint_path)


if __name__ == '__main__':
  attack = Attack()
  attack.run()
