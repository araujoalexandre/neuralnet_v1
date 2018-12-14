import random
from os.path import join
import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
from tensorflow import flags

from config import hparams as FLAGS

class MNISTReader:

  def __init__(self, batch_size, num_epochs=1, is_training=False):

    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.is_training = is_training
    self.height, self.width = 28, 28
    self.n_train_files = 60000
    self.n_test_files = 10000
    self.n_classes = 10
    self.batch_shape = (None, 32, 32, 1)

    readers_params = FLAGS.readers_params
    self.num_parallel_calls = readers_params['num_parallel_calls']
    self.num_parallel_readers = readers_params['num_parallel_readers']
    self.prefetch_buffer_size = readers_params['prefetch_buffer_size']

    data = FLAGS.data
    self.files = gfile.Glob(join(data['data_dir'], 'mnist',
                                 data['data_pattern']))
    if not self.files:
      raise IOError("Unable to find files in data_dir '{}'.".format(
        data['data_dir']))
    logging.info("Number of training TFRecord files: {}.".format(
      len(self.files)))

  def _image_preprocessing(self, image_buffer):
    """Decode and preprocess one image for evaluation or training.
    Args:
      image_buffer: JPEG encoded string Tensor
    Returns:
      Tensor containing an appropriately scaled image
    """
    image = tf.decode_raw(image_buffer, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, (1, self.height, self.width, 1))
    image = tf.pad(image, ((0,0), (2,2), (2,2), (0,0)), mode='constant')
    _, self.height, self.width, _ = image.get_shape().as_list()
    image = tf.reshape(image, (self.height, self.width, 1))
    image = tf.divide(image, 255)
    return image

  def _parse_fn(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers.
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
    """
    feature_map = {
      'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/height': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/width': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
    }
    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/label'], dtype=tf.int32)
    return features['image'], label

  def _parse_and_processed(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    """
    image, label = self._parse_fn(example_serialized)
    image = self._image_preprocessing(image)
    return image, label

  def input_fn(self):
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
      with tf.name_scope('batch_processing'):
        files = tf.data.Dataset.list_files(self.files)
        dataset = files.apply(tf.data.experimental.parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=self.num_parallel_readers))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=self._parse_and_processed, batch_size=self.batch_size,
            num_parallel_calls=self.num_parallel_calls))
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
        if self.is_training:
          dataset = dataset.shuffle(buffer_size=3*self.batch_size)
          dataset = dataset.repeat(self.num_epochs)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
    # Display the training images in the visualizer.
    tf.summary.image('images', image_batch)
    return image_batch, label_batch


class CIFAR10Reader:

  def __init__(self, batch_size, num_epochs=1, is_training=False,
    *args, **kwargs):

    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.is_training = is_training
    self.height, self.width = 32, 32
    self.n_train_files = 50000
    self.n_test_files = 10000
    self.n_classes = 10
    self.batch_shape = (None, 32, 32, 3)

    readers_params = FLAGS.readers_params
    self.num_parallel_calls = readers_params['num_parallel_calls']
    self.num_parallel_readers = readers_params['num_parallel_readers']
    self.prefetch_buffer_size = readers_params['prefetch_buffer_size']

    data = FLAGS.data
    self.files = gfile.Glob(join(data["data_dir"], 'cifar10',
                                 data["data_pattern"]))
    if not self.files:
      raise IOError("Unable to find training files. data_pattern='{}'.".format(
        data["data_pattern"]))
    logging.info("Number of training TFRecord files: {}.".format(
      len(self.files)))

  def _image_preprocessing(self, image_buffer):
    """Decode and preprocess one image for evaluation or training.
    Args:
      image: image as numpy array
    Returns:
      3D Tensor containing an appropriately scaled image
    """
    # Decode the string as an RGB JPEG.
    image = tf.decode_raw(image_buffer, tf.uint8)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image, (3, self.height, self.width))
    image = tf.transpose(image, [1, 2, 0])
    image = tf.image.per_image_standardization(image)
    return image

  def _parse_fn(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers.
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
    """
    feature_map = {
      'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/height': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/width': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value='')
    }
    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/label'], dtype=tf.int32)
    return features['image'], label

  def _parse_and_processed(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    """
    image, label = self._parse_fn(example_serialized)
    image = self._image_preprocessing(image)
    return image, label

  def input_fn(self):
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
      with tf.name_scope('batch_processing'):
        files = tf.data.Dataset.list_files(self.files)
        dataset = files.apply(tf.data.experimental.parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=self.num_parallel_readers))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=self._parse_and_processed, batch_size=self.batch_size,
            num_parallel_calls=self.num_parallel_calls))
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
        if self.is_training:
          dataset = dataset.shuffle(buffer_size=3*self.batch_size)
          dataset = dataset.repeat(self.num_epochs)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
    # Display the training images in the visualizer.
    tf.summary.image('images', image_batch)
    return image_batch, label_batch
