import random
from os.path import join

import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
from tensorflow import flags
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.contrib.data.python.ops import threadpool

from config import hparams as FLAGS

class BaseReader:

  def _get_tfrecords(self, name):
    paths = list(map(lambda x: join(FLAGS.data_dir, name, x),
                     FLAGS.data_pattern.split(',')))
    files = gfile.Glob(paths)
    if not files:
      raise IOError("Unable to find files. data_pattern='{}'.".format(
        FLAGS.data_pattern))
    logging.info("Number of TFRecord files: {}.".format(
      len(files)))
    return files

  def _maybe_one_hot_encode(self, labels):
    """One hot encode the labels"""
    if FLAGS.one_hot_labels:
      labels = tf.one_hot(labels, self.n_classes)
      labels = tf.squeeze(labels)
      return labels
    return labels

  def _parse_and_preprocess(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    """
    image, label = self._parse_fn(example_serialized)
    image = self._image_preprocessing(image)
    label = self._maybe_one_hot_encode(label)
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
    files = tf.constant(self.files, name="tfrecord_files")
    with tf.name_scope('batch_processing'):
      ds = tf.data.TFRecordDataset.list_files(files, shuffle=shuffle)
      ds = ds.apply(
        tf.data.experimental.parallel_interleave(
          tf.data.TFRecordDataset,
          cycle_length=FLAGS.datasets_parallel_interleave_cycle_length or 10,
          sloppy=FLAGS.datasets_sloppy_parallel_interleave,
          prefetch_input_elements=FLAGS.datasets_parallel_interleave_prefetch))
      ds = ds.prefetch(buffer_size=self.batch_size)
      if FLAGS.datasets_use_caching:
        ds = ds.cache()
      if self.is_training:
        ds = ds.apply(
          tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
      else:
        ds = ds.repeat()
      ds = ds.map(self._parse_and_preprocess,
              num_parallel_calls=self.num_parallel_calls)
      ds = ds.batch(self.batch_size_per_split,
            drop_remainder=self.drop_remainder)
      ds = ds.prefetch(buffer_size=self.num_splits)
      logging.info('num_threads {}'.format(self.num_threads))
      if self.num_threads:
        ds = threadpool.override_threadpool(
          ds,
          threadpool.PrivateThreadPool(
            self.num_threads, display_name='input_pipeline_thread_pool'))
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          ds,
          self.gpu_devices,
          source_device=self.cpu_device)
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           multi_device_iterator.initializer)
      return multi_device_iterator


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

    self.files = self._get_tfrecords('mnist')

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

    self.files = self._get_tfrecords('fashion_mnist')

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

    self.files = self._get_tfrecords('cifar10')

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
    image = tf.reshape(image, (self.height, self.width, 3))
    if self.use_gray_scale:
      image = tf.image.rgb_to_grayscale(image)
      image = tf.reshape(image, (1, self.height * self.width))
    image = tf.cast(image, dtype=tf.float32)
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

    self.files = self._get_tfrecords('cifar100')

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

    self.files = self._get_tfrecords('yt8m/video')

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

    self.files = self._get_tfrecords('yt8m/frame')

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


class IMAGENETReader(BaseReader):

  def __init__(self, batch_size, gpu_devices, cpu_device,
               is_training, *args, **kwargs):

    # Provide square images of this size. 
    self.image_size = FLAGS.image_size

    self.gpu_devices = gpu_devices
    self.cpu_device = cpu_device
    self.num_splits = len(gpu_devices)
    self.batch_size = batch_size
    self.batch_size_per_split = batch_size // self.num_splits
    self.is_training = is_training
    self.height, self.width = self.image_size, self.image_size
    self.n_train_files = 1281167
    if FLAGS.readers_params['drop_remainder']:
      # remove reminder from n_train_files
      n_step_by_epoch = self.n_train_files // self.batch_size
      self.n_train_files = n_step_by_epoch * self.batch_size

    self.n_test_files = 50000
    self.n_classes = 1001
    self.batch_shape = (None, self.height, self.height, 1)
    self.display_tensorboard = FLAGS.display_tensorboard

    readers_params = FLAGS.readers_params
    self.num_threads = FLAGS.datasets_num_private_threads
    self.num_parallel_calls = readers_params['num_parallel_calls']
    self.num_parallel_readers = readers_params['num_parallel_readers']
    self.prefetch_buffer_size = readers_params['prefetch_buffer_size']

    self.files = self._get_tfrecords('imagenet')

  def _parse_and_preprocess(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    """
    image_buffer, label, bbox, _ = self._parse_fn(example_serialized)
    image = self._image_preprocessing(image_buffer, bbox)
    return image, label

  def _parse_fn(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:
      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox, features['image/class/text']

  def _decode_jpeg(self, image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_jpeg'):
      # Decode the string as an RGB JPEG.
      # Note that the resulting image contains an unknown height and width
      # that is set dynamically by decode_jpeg. In other words, the height
      # and width of image is unknown at compile-time.
      image = tf.image.decode_jpeg(image_buffer, channels=3)

      # After this point, all image pixels reside in [0,1)
      # until the very end, when they're rescaled to (-1, 1).  The various
      # adjust_* ops all require this range for dtype float.
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      return image

  def _eval(self, image, scope=None):
    """Prepare one image for evaluation.
    Args:
      image: 3-D float Tensor
      height: integer
      width: integer
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    height, width = self.height, self.width
    with tf.name_scope(values=[image, height, width], name=scope,
                       default_name='eval_image'):
      # Crop the central region of the image with an area containing 87.5% of
      # the original image.
      image = tf.image.central_crop(image, central_fraction=0.875)

      # Resize the image to the original height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
      return image

  def _distort_color(self, image, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. We randomly permute the ordering of the color ops.
    Args:
      image: Tensor containing single image.
      scope: Optional scope for name_scope.
    Returns:
      color-distorted image
    """
    with tf.name_scope(values=[image], name=scope, default_name='distort_color'):

      def color_ordering_1(image):
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        return image

      def color_ordering_2(image):
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        return image

      rand = tf.random_uniform([1], 0, 2, tf.int32)
      image = tf.cond(tf.equal(rand[0], 1),
                      lambda: color_ordering_1(image),
                      lambda: color_ordering_2(image))

      # The random_* ops do not necessarily clamp.
      image = tf.clip_by_value(image, 0.0, 1.0)
      return image

  def _data_augmentation(self, image, bbox, scope=None):
    """Distort one image for training a network.
    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.
    Args:
      image: 3-D float Tensor of image
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    height, width = self.height, self.width
    with tf.name_scope(values=[image, height, width, bbox], name=scope,
                       default_name='distort_image'):
      # Each bounding box has shape [1, num_boxes, box coords] and
      # the coordinates are ordered [ymin, xmin, ymax, xmax].

      # Display the bounding box in the first thread only.
      if self.display_tensorboard:
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                      bbox)
        tf.summary.image('image_with_bounding_boxes', image_with_box)

      # A large fraction of image datasets contain a human-annotated bounding
      # box delineating the region of the image containing the object of interest.
      # We choose to create a new bounding box for the object which is a randomly
      # distorted version of the human-annotated bounding box that obeys an 
      # allowed range of aspect ratios, sizes and overlap with the human-annotated
      # bounding box. If no box is supplied, then we assume the bounding box is
      # the entire image.
      sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bbox,
          min_object_covered=0.1,
          aspect_ratio_range=[0.75, 1.33],
          area_range=[0.05, 1.0],
          max_attempts=100,
          use_image_if_no_bounding_boxes=True)
      bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
      if self.display_tensorboard:
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distort_bbox)
        tf.summary.image('images_with_distorted_bounding_box',
                         image_with_distorted_box)

      # Crop the image to the specified bounding box.
      distorted_image = tf.slice(image, bbox_begin, bbox_size)

      # This resizing operation may distort the images because the aspect
      # ratio is not respected. We select a resize method randomly 
      # Note that ResizeMethod contains 4 enumerated resizing methods.
      # resize_method = tf.random_uniform([1], 0, 4, tf.int32)
      # distorted_image = tf.image.resize_images(distorted_image, [height, width],
      #                                          method=resize_method[0])

      distorted_image = tf.image.resize_images(distorted_image, [height, width],
                                               method=0)

      # Restore the shape since the dynamic slice based upon the bbox_size loses
      # the third dimension.
      distorted_image.set_shape([height, width, 3])
      if self.display_tensorboard:
        tf.summary.image('cropped_resized_image',
                         tf.expand_dims(distorted_image, 0))

      # Randomly flip the image horizontally.
      distorted_image = tf.image.random_flip_left_right(distorted_image)

      # Randomly distort the colors.
      distorted_image = self._distort_color(distorted_image)

      if self.display_tensorboard:
        tf.summary.image('final_distorted_image',
                         tf.expand_dims(distorted_image, 0))
      return distorted_image

  def _image_preprocessing(self, image_buffer, bbox):
    """Decode and preprocess one image for evaluation or training.
    Args:
      image: image as numpy array
    Returns:
      3D Tensor containing an appropriately scaled image
    """
    # Decode the string as an RGB JPEG.
    image = self._decode_jpeg(image_buffer)

    if self.is_training:
      image = self._data_augmentation(image, bbox)
    else:
      image = self._eval(image)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

