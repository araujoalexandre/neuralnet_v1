
import os
import sys
import six
import random
import tarfile
import pickle
import logging
import shutil
from os.path import join, exists
from six.moves import urllib
from datetime import datetime
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow.python_io import TFRecordWriter
from tensorflow.keras import datasets

sys.path.insert(0, "../models")
from scattering_utils import Scattering
from readers import readers_config


FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "/tmp/",
                    "Output data directory")

flags.DEFINE_string("dataset", "",
                    "Dataset to save as TFRecords.")

flags.DEFINE_integer("train_split", 10,
                     "Number of TFRecords to split the train dataset")

flags.DEFINE_integer("test_split", 1,
                     "Number of TFRecords to split the test dataset")


flags.DEFINE_string("data_dir", "",
                    "Directory of ImageNet TFRecords.")

flags.DEFINE_integer("J", 2,
                     "Params for Scattering Transform.")

flags.DEFINE_integer("imagenet_image_size", 299,
                     "Size of ImageNet images.")



def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(value, six.text_type):
    value = six.binary_type(value, encoding='utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



class ConvertDataset:

  def reshape(self):
    if self.x_train.ndim == 3:
      _, self.height, self.width = self.x_train.shape
      dim = self.height * self.width
    else:
      _, self.height, self.width, self.channels = self.x_train.shape
      dim = self.height * self.width * self.channels
    self.x_train = self.x_train.reshape(-1, dim)
    self.x_test = self.x_test.reshape(-1, dim)
    self.y_train = self.y_train.reshape(-1, 1)
    self.y_test = self.y_test.reshape(-1, 1)

  def _convert_to_example(self, image, label):
    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(image),
      'image/height': _int64_feature(self.height),
      'image/width': _int64_feature(self.width),
      'image/label': _int64_feature(label)}))
    return example

  def _process_images(self, name, images, labels, id_file, n_files):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      name: string, unique identifier specifying the data set
      images: array of images
      labels: array of labels
    """
    output_filename = '{}-{:05d}-of-{:05d}'.format(name, id_file, n_files)
    output_file = os.path.join(FLAGS.output_dir, self.name, output_filename)
    with TFRecordWriter(output_file) as writer:
      for image, label in zip(images, labels):
        example = self._convert_to_example(image.tobytes(), label)
        writer.write(example.SerializeToString())
    print('{}: Wrote {} images to {}'.format(
        datetime.now(), len(images), output_file), flush=True)

  def split_data(self, x, y, n_split):
    x_split = np.array_split(x, n_split)
    y_split = np.array_split(y, n_split)
    return x_split, y_split

  def convert(self):
    """Main method to convert mnist images to TFRecords
    """
    data_folder = join(FLAGS.output_dir, self.name)
    if exists(data_folder):
      logging.info('{} data already converted to TFRecords.'.format(self.name))
      return
    os.makedirs(data_folder)
    x_train_split, y_train_split = self.split_data(
      self.x_train, self.y_train, FLAGS.train_split)
    x_test_split, y_test_split = self.split_data(
      self.x_test, self.y_test, FLAGS.test_split)
    for i, (x, y) in enumerate(zip(x_train_split, y_train_split), 1):
      self._process_images("train", x, y, i, FLAGS.train_split)
    for i, (x, y) in enumerate(zip(x_test_split, y_test_split), 1):
      self._process_images("test", x, y, i , FLAGS.test_split)


class ConvertMNIST(ConvertDataset):
  def __init__(self):
    self.name = "mnist"
    (self.x_train, self.y_train), (self.x_test, self.y_test) = \
        datasets.mnist.load_data()
    self.reshape()

class ConvertFashionMNIST(ConvertDataset):
  def __init__(self):
    self.name = "fashion_mnist"
    (self.x_train, self.y_train), (self.x_test, self.y_test) = \
        datasets.fashion_mnist.load_data()
    self.reshape()

class ConvertCIFAR10(ConvertDataset):
  def __init__(self):
    self.name = "cifar10"
    (self.x_train, self.y_train), (self.x_test, self.y_test) = \
        datasets.cifar10.load_data()
    self.reshape()

class ConvertCIFAR100(ConvertDataset):
  def __init__(self):
    self.name = "cifar100"
    (self.x_train, self.y_train), (self.x_test, self.y_test) = \
        datasets.cifar100.load_data()
    self.reshape()








class ConvertCIFAR10Scattering(ConvertDataset):
  def __init__(self, J=2):
    self.J = J
    self.name = "cifar10_scattering_j{}".format(self.J)
    (self.x_train, self.y_train), (self.x_test, self.y_test) = \
        datasets.cifar10.load_data()
    self.x_train = self.make_scattering(self.x_train, J=1)
    self.x_test = self.make_scattering(self.x_test, J=1)
    self.reshape()

  def make_scattering(self, x, J=None):
    """Transform images with Scattering Transform."""
    assert J != None
    with tf.device('/gpu:0'):
      x_placeholder = tf.placeholder(
        tf.float32, shape=(None, 3, 32, 32))
      scattering_transform = Scattering(M=M, N=N, J=2)
      x_scattering = scattering_transform(x_placeholder)

    x = x / 255
    x_scattering = []
    batch_size = 100
    n_batch = x.shape[0] / batch_size
    x_splited = np.array_split(x, n_batch)
    with tf.Session() as sess:
      for batch in x_splited:
        data_to_feed = {x_placeholder: batch}
        x_scattering_ = sess.run(x_scattering, feed_dict=data_to_feed)
        x_scattering.append(x_scattering_)
    return np.vstack(x_scattering)




class Params:
  pass

class ConvertImageNetScattering(ConvertDataset):
  def __init__(self):
    self.J = FLAGS.J
    self.name = "imagenet_{}_scattering_j{}".format(
      FLAGS.imagenet_image_size, self.J)

    output_dir = join(FLAGS.output_dir, self.name)
    # if exists(output_dir):
    #   shutil.rmtree(output_dir)
    if not exists(output_dir):
      os.mkdir(output_dir)

    self.params = Params()
    self.params.imagenet_image_size = FLAGS.imagenet_image_size
    self.params.data_dir = FLAGS.data_dir
    self.params.one_hot_labels = False
    self.params.summary_verbosity = None
    self.params.datasets_num_private_threads = 10
    self.params.datasets_interleave_cycle_length = 10
    self.params.datasets_interleave_block_length = 10
    self.params.datasets_use_caching = False

    self.height = self.params.imagenet_image_size
    self.width = self.params.imagenet_image_size


  def convert(self):
    if self.J != 1:
      self.make_scattering("train", self.params)
    self.make_scattering("valid", self.params)

  def make_scattering(self, dataset, params):
    """Transform images with Scattering Transform."""
    params.data_pattern = "{}*".format(dataset)
    reader = readers_config['imagenet'](
      params, 16, ['/gpu:0'],
      '/cpu:0', is_training=False)
    if dataset == "train":
      n_images = getattr(reader, 'n_train_files')
    else:
      n_images = getattr(reader, 'n_test_files')
    n_files = n_images // 1024
    img_size = params.imagenet_image_size
    images, labels = reader.input_fn().get_next()[0]
    images = tf.transpose(images, [0, 3, 1, 2])
    scattering_transform = Scattering(
      M=img_size, N=img_size, J=self.J)
    images = scattering_transform(images)

    local_var_init_op = tf.local_variables_initializer()
    table_init_ops = tf.tables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    if table_init_ops:
      variable_mgr_init_ops.extend([table_init_ops])
    local_var_init_op_group = tf.group(*variable_mgr_init_ops)

    batch_images, batch_labels = [], []
    with tf.Session() as sess:
      sess.run(local_var_init_op_group)
      id_file, counter = 0, 0
      while True:
        try:
          images_, labels_ = sess.run([images, labels])
          batch_images.append(images_)
          batch_labels.append(labels_)
          counter += 1
          if counter == 64:
            batch_images = np.vstack(batch_images)
            batch_size, *feature_shape = batch_images.shape
            batch_images = batch_images.reshape(batch_size, -1)
            batch_labels = np.vstack(batch_labels)
            print("saving batch {}".format(id_file))
            self._process_images(
              dataset, batch_images, batch_labels, id_file, n_files)
            batch_images, batch_labels = [], []
            counter = 0
            id_file += 1
        except tf.errors.OutOfRangeError:
          break
    return


def main(_):
  if FLAGS.dataset == "mnist":
    ConvertMNIST().convert()
  elif FLAGS.dataset == "fashion_mnist":
    ConvertFashionMNIST().convert()
  elif FLAGS.dataset == "cifar10":
    ConvertCIFAR10().convert()
  elif FLAGS.dataset == "cifar_scattering":
    ConvertCIFAR10Scattering().convert()
  elif FLAGS.dataset == "cifar100":
    ConvertCIFAR100().convert()
  elif FLAGS.dataset == "imagenet_scattering":
    ConvertImageNetScattering().convert()
  elif FLAGS.dataset == "all":
    ConvertMNIST().convert()
    ConvertFashionMNIST().convert()
    ConvertCIFAR10().convert()
    ConvertCIFAR100().convert()


if __name__ == '__main__':
  tf.app.run()
