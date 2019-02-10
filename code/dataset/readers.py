import random
from os.path import join
import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
from tensorflow import flags

from config import hparams as FLAGS

class BaseReader:

  def _maybe_one_hot_encode(self, labels):
    """One hot encode the labels"""
    if FLAGS.one_hot_labels:
      labels = tf.one_hot(labels, self.n_classes)
      labels = tf.squeeze(labels)
      return labels
    return labels

  def _parse_and_processed(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    """
    image, label = self._parse_fn(example_serialized)
    image = self._image_preprocessing(image)
    return image, label

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

  def input_fn(self):

    config = FLAGS.readers_params
    self.cache_dataset = config['cache_dataset']
    self.drop_remainder = config['drop_remainder']

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    shuffle = True if self.is_training else False
    sloppy = True if self.is_training else False
    with tf.device('/cpu:0'):
      files = tf.constant(self.files, name="tfrecord_files")
      with tf.name_scope('batch_processing'):
        dataset = tf.data.TFRecordDataset(files,
                        num_parallel_reads=self.num_parallel_readers)
        dataset = dataset.map(self._parse_and_processed,
                          num_parallel_calls=self.num_parallel_calls)
        if self.is_training:
          dataset = dataset.shuffle(buffer_size=5*self.batch_size)
        dataset = dataset.batch(self.batch_size,
                                drop_remainder=self.drop_remainder)
        if self.is_training:
          dataset = dataset.repeat()
          if self.cache_dataset:
            dataset = dataset.cache()
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
    label_batch = self._maybe_one_hot_encode(label_batch)
    return image_batch, label_batch


class MNISTReader(BaseReader):

  def __init__(self, batch_size, is_training, *args, **kwargs):

    self.batch_size = batch_size
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

    self.files = gfile.Glob(join(
      FLAGS.data_dir, 'mnist', FLAGS.data_pattern))

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
    if FLAGS.per_image_standardization:
      image = tf.image.per_image_standardization(image)
    else:
      image = (image / 255 - 0.5) * 2
    return image


class FashionMNISTReader(BaseReader):

  def __init__(self, batch_size, is_training, *args, **kwargs):

    self.batch_size = batch_size
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

    self.files = gfile.Glob(join(
      FLAGS.data_dir, 'fashion_mnist', FLAGS.data_pattern))

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
    if FLAGS.per_image_standardization:
      image = tf.image.per_image_standardization(image)
    else:
      image = (image / 255 - 0.5) * 2
    return image


class CIFAR10Reader(BaseReader):

  def __init__(self, batch_size, is_training, *args, **kwargs):

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 32, 32
    self.n_train_files = 50000
    self.n_test_files = 10000
    self.n_classes = 10
    self.batch_shape = (None, 32, 32, 3)
    self.use_data_augmentation = FLAGS.data_augmentation
    self.use_gray_scale = FLAGS.grayscale

    # use grey scale
    if self.use_gray_scale:
      self.batch_shape = (None, 32, 32, 1)

    readers_params = FLAGS.readers_params
    self.num_parallel_calls = readers_params['num_parallel_calls']
    self.num_parallel_readers = readers_params['num_parallel_readers']
    self.prefetch_buffer_size = readers_params['prefetch_buffer_size']

    self.files = gfile.Glob(join(
      FLAGS.data_dir, 'cifar10', FLAGS.data_pattern))

    if not self.files:
      raise IOError("Unable to find training files. data_pattern='{}'.".format(
        data["data_pattern"]))
    logging.info("Number of training TFRecord files: {}.".format(
      len(self.files)))

  def _data_augmentation(self, image):
    image = tf.image.resize_image_with_crop_or_pad(
                        image, self.height+4, self.width+4)
    image = tf.image.random_crop(image, [self.height, self.width, 3])
    image = tf.image.random_flip_left_right(image)
    return image

  def _image_preprocessing(self, image_buffer):
    """Decode and preprocess one image for evaluation or training.
    Args:
      image: image as numpy array
    Returns:
      3D Tensor containing an appropriately scaled image
    """
    # Decode the string as an RGB JPEG.
    image = tf.decode_raw(image_buffer, tf.uint8)
    if self.use_gray_scale:
      image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image, (self.height, self.width, 3))
    if self.use_data_augmentation and self.is_training:
      image = self._data_augmentation(image)
    if FLAGS.per_image_standardization:
      image = tf.image.per_image_standardization(image)
    elif FLAGS.dataset_standardization:
      mean = [125.3, 123.0, 113.9]
      std  = [63.0,  62.1,  66.7]
      image = (image - mean) / std
    else:
      image = (image / 255 - 0.5) * 2
    return image


class CIFAR100Reader(BaseReader):

  def __init__(self, batch_size, is_training, *args, **kwargs):

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 32, 32
    self.n_train_files = 50000
    self.n_test_files = 10000
    self.n_classes = 100
    self.batch_shape = (None, 32, 32, 3)
    self.use_data_augmentation = FLAGS.data_augmentation
    self.use_gray_scale = FLAGS.grayscale

    if self.use_gray_scale:
      self.batch_shape = (None, 32, 32, 1)

    readers_params = FLAGS.readers_params
    self.num_parallel_calls = readers_params['num_parallel_calls']
    self.num_parallel_readers = readers_params['num_parallel_readers']
    self.prefetch_buffer_size = readers_params['prefetch_buffer_size']

    self.files = gfile.Glob(join(
      FLAGS.data_dir, 'cifar100', FLAGS.data_pattern))

    if not self.files:
      raise IOError("Unable to find training files. data_pattern='{}'.".format(
        data["data_pattern"]))
    logging.info("Number of training TFRecord files: {}.".format(
      len(self.files)))

  def _data_augmentation(self, image):
    image = tf.image.resize_image_with_crop_or_pad(
                        image, self.height+4, self.width+4)
    image = tf.image.random_crop(image, [self.height, self.width, 3])
    image = tf.image.random_flip_left_right(image)
    return image

  def _image_preprocessing(self, image_buffer):
    """Decode and preprocess one image for evaluation or training.
    Args:
      image: image as numpy array
    Returns:
      3D Tensor containing an appropriately scaled image
    """
    # Decode the string as an RGB JPEG.
    image = tf.decode_raw(image_buffer, tf.uint8)
    if self.use_gray_scale:
      image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image, (self.height, self.width, 3))
    if self.use_data_augmentation and self.is_training:
      image = self._data_augmentation(image)
    if FLAGS.per_image_standardization:
      image = tf.image.per_image_standardization(image)
    else:
      image = (image / 255 - 0.5) * 2
    return image


class YT8MAggregatedFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """
  def __init__(self, batch_size, is_training, *args, **kwargs):

    self.batch_size = batch_size
    self.is_training = is_training
    self.n_train_files = 3888919
    self.n_test_files = 1112356 # public
    # self.n_test_files = 1133323 # private
    self.n_classes = 3862
    self.batch_shape = (None, 1152)

    self.feature_sizes = [1024, 128]
    self.feature_names = ["mean_rgb", "mean_audio"]

    readers_params = FLAGS.readers_params
    self.num_parallel_calls = readers_params['num_parallel_calls']
    self.num_parallel_readers = readers_params['num_parallel_readers']
    self.prefetch_buffer_size = readers_params['prefetch_buffer_size']

    self.files = gfile.Glob(join(
      FLAGS.data_dir, 'yt8m', 'video', FLAGS.data_pattern))

    if not self.files:
      raise IOError("Unable to find training files. data_pattern='{}'.".format(
        FLAGS.data_pattern))
    logging.info("Number of training TFRecord files: {}.".format(
      len(self.files)))

  def _image_preprocessing(self, video):
    # no preprocessing done
    return video

  def _parse_fn(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
    """
    feature_map = {"id": tf.FixedLenFeature([], tf.string),
                   "labels": tf.VarLenFeature(tf.int64),
                   "mean_rgb": tf.FixedLenFeature(1024, tf.float32),
                   "mean_audio": tf.FixedLenFeature(128, tf.float32)}

    features = tf.parse_single_example(example_serialized, features=feature_map)
    # features = tf.parse_example(example_serialized, features=feature_map)
    labels = tf.sparse_to_indicator(features["labels"], self.n_classes)
    concatenated_features = tf.concat([
        features[name] for name in self.feature_names], 0)
    return concatenated_features, labels


class YT8MFrameFeatureReader(BaseReader):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(self, batch_size, is_training, *args, **kwargs):

    self.batch_size = batch_size
    self.is_training = is_training
    self.n_train_files = 3888919
    self.n_test_files = 1112356 # public
    # self.n_test_files = 1133323 # private
    self.n_classes = 3862
    self.batch_shape = (None, 300, 1152)

    self.feature_sizes = [1024, 128]
    self.feature_names = ["rgb", "audio"]
    self.max_frames = 300

    self.max_quantized_value = 2
    self.min_quantized_value = -2

    readers_params = FLAGS.readers_params
    self.num_parallel_calls = readers_params['num_parallel_calls']
    self.num_parallel_readers = readers_params['num_parallel_readers']
    self.prefetch_buffer_size = readers_params['prefetch_buffer_size']

    self.files = gfile.Glob(join(
      FLAGS.data_dir, 'yt8m', 'frame', FLAGS.data_pattern))

    if not self.files:
      raise IOError("Unable to find training files. data_pattern='{}'.".format(
        FLAGS.data_pattern))
    logging.info("Number of training TFRecord files: {}.".format(
      len(self.files)))

  def get_video_matrix(self, features, feature_size):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], self.max_frames)
    feature_matrix = Dequantize(decoded_features,
                                      self.max_quantized_value,
                                      self.min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, self.max_frames)
    return feature_matrix, num_frames

  def _image_preprocessing(self, video):
    # no preprocessing
    return video

  def _parse_fn(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
    """
    num_features = len(self.feature_names)
    context_features = {"id": tf.FixedLenFeature([], tf.string),
                        "labels": tf.VarLenFeature(tf.int64)}
    sequence_features={
        feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self.feature_names}

    contexts, features = tf.parse_single_sequence_example(
        example_serialized,
        context_features=context_features,
        sequence_features=sequence_features)

    # read ground truth labels
    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (self.n_classes,), 1,
            validate_indices=False),
        tf.bool))

    feature_matrices = [None] * num_features  # an array of different features
    for feature_index in range(num_features):
      feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
          features[self.feature_names[feature_index]],
          self.feature_sizes[feature_index])
      feature_matrices[feature_index] = feature_matrix

    # concatenate different features
    video_matrix = tf.concat(feature_matrices, 1)
    return video_matrix, labels


