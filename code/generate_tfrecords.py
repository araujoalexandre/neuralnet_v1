
import os, sys, six
import random
import tarfile
import pickle
from os.path import join, exists
from six.moves import urllib
from datetime import datetime
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python_io import TFRecordWriter

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "/tmp/",
                    "Output data directory")


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


class ConvertMNIST:

  def __init__(self):
    self.height, self.width = 28, 28
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    self.x_train, self.x_test = x_train, x_test
    self.y_train, self.y_test = y_train, y_test

  def _convert_to_example(self, image, label):
    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(image),
      'image/height': _int64_feature(self.height),
      'image/width': _int64_feature(self.width),
      'image/label': _int64_feature(label)}))
    return example

  def _process_images(self, name, images, labels):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      name: string, unique identifier specifying the data set
      images: array of images
      labels: array of labels
    """
    output_filename = '{}-00001-of-00001'.format(name)
    output_file = os.path.join(FLAGS.output_dir, 'mnist', output_filename)
    with TFRecordWriter(output_file) as writer:
      for image, label in zip(images, labels):
        example = self._convert_to_example(image.tobytes(), label)
        writer.write(example.SerializeToString())
    print('{}: Wrote {} images to {}'.format(
        datetime.now(), len(images), output_file), flush=True)

  def convert(self):
    """Main method to convert mnist images to TFRecords
    """
    data_folder = join(FLAGS.output_dir, 'mnist')
    if exists(data_folder):
      logging.info('mnist data already converted to TFRecords.')
      return 
    os.makedirs(data_folder)
    self._process_images("train", self.x_train, self.y_train)
    self._process_images("test", self.x_test, self.y_test)


class ConvertCIFAR:

  def __init__(self):
    
    self.url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    self.maybe_download_and_extract()
    self.height, self.width = 32, 32
    self.num_classes = 10
    self.train_size = 50000
    self.test_size = 10000

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
    self._sess = tf.Session()

  def maybe_download_and_extract(self):
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.output_dir
    if not exists(dest_directory):
      os.makedirs(dest_directory)
    filename = self.url.split('/')[-1]
    filepath = join(dest_directory, filename)
    if not exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(self.url, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      logging.info('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    self.extracted_dir_path = join(dest_directory, 'cifar-10-batches-py')
    if not exists(self.extracted_dir_path):
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    tf.gfile.Remove(filepath)
  
  def _unpickle(self, file):
      with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
      return data

  def _convert_to_example(self, image, label, filename):
    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(image),
      'image/height': _int64_feature(self.height),
      'image/width': _int64_feature(self.width),
      'image/label': _int64_feature(label),
      'image/filename': _bytes_feature(filename)}))
    return example

  def _decode_jpeg(self, image_data):
    image = self._sess.run(
      self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def _process_images(self, name, images, labels, filenames, id_file, nfiles):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      name: string, unique identifier specifying the data set
      images: array of images
      labels: array of labels
    """
    output_filename = '{}-{:05d}-of-{:05d}'.format(name, id_file, nfiles)
    output_file = join(FLAGS.output_dir, 'cifar10', 
      output_filename)
    with TFRecordWriter(output_file) as writer:
      for image, label, filename in zip(images, labels, filenames):
        example = self._convert_to_example(image.tobytes(), label, filename)
        writer.write(example.SerializeToString())
    print('{}: Wrote {} images to {}'.format(
        datetime.now(), len(images), output_file), flush=True)    

  def convert(self):
    """Main method to convert cifar images to TFRecords
    """
    data_folder = join(FLAGS.output_dir, 'cifar10')
    if exists(data_folder):
      logging.info('cifar10 data already converted to TFRecords.')
      return 
    os.makedirs(data_folder)
    train_files = tf.gfile.Glob(join(self.extracted_dir_path, 'data*'))
    test_file = tf.gfile.Glob(join(self.extracted_dir_path, 'test_batch'))[0]
    nfiles = len(train_files)
    for i, train_file in enumerate(train_files, 1):
      data = self._unpickle(train_file)
      images, labels, filenames = data[b'data'], data[b'labels'], data[b'filenames']
      self._process_images('train', images, labels, filenames, i, nfiles)
    data = self._unpickle(test_file)
    images, labels, filename = data[b'data'], data[b'labels'], data[b'filenames']
    self._process_images('test', images, labels, filenames, 1, 1)
    tf.gfile.DeleteRecursively(self.extracted_dir_path)


def main(_):
  ConvertMNIST().convert()
  ConvertCIFAR().convert()

if __name__ == '__main__':
  tf.app.run()
