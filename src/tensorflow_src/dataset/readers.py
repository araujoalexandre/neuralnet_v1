
import logging
from os.path import join
from functools import partial

import tensorflow as tf
from tensorflow.io import gfile
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.contrib.data.python.ops import threadpool

from . import autoaugment


def normalize_image(image):
  # rescale between [-1, 1]
  image = tf.multiply(image, 1. / 127.5)
  return tf.subtract(image, 1.0)


class MultiDeviceIterator:

  def __init__(self, iterator, gpu_devices):
    self.iterator = iterator
    self.gpu_devices = gpu_devices

  def get_next(self):
    l = []
    for device in self.gpu_devices:
      with tf.device(device):
        l.append(self.iterator.get_next())
    return l



class BaseReader:

  def __init__(self, params, batch_size, gpu_devices, cpu_device,
               is_training):
    self.params = params
    self.gpu_devices = gpu_devices
    self.cpu_device = cpu_device
    self.num_splits = len(gpu_devices)
    self.batch_size = batch_size
    self.batch_size_per_split = batch_size // self.num_splits
    self.is_training = is_training
    self.summary_verbosity = self.params.summary_verbosity

    self.num_threads = self.params.reader['num_private_threads']
    self.interleave_cycle_length = \
        self.params.reader['interleave_cycle_length']
    self.interleave_block_length = \
        self.params.reader['interleave_block_length']
    self.use_caching = self.params.reader['use_caching']
    self.shuffle_buffer_size = self.params.reader['shuffle_buffer_size']

  def _get_tfrecords(self, name):
    paths = self.params.data_dir.split(':')
    data_dir = None
    for path in paths:
      if gfile.exists(join(path, name)):
        data_dir = path
        break
    assert data_dir is not None, "data_dir not found"
    paths = list(map(lambda x: join(data_dir, name, x),
                     self.params.data_pattern.split(',')))
    files = gfile.glob(paths)
    if not files:
      raise IOError("Unable to find files. data_pattern='{}'.".format(
        self.params.data_pattern))
    logging.info("Number of TFRecord files: {}.".format(
      len(files)))
    return files

  def _maybe_one_hot_encode(self, labels):
    """One hot encode the labels"""
    if self.params.reader['one_hot_labels']:
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
    shuffle = True if self.is_training else False
    sloppy = True if self.is_training else False
    files = tf.constant(self.files, name="tfrecord_files")
    with tf.name_scope('batch_processing'):
      ds = tf.data.TFRecordDataset.list_files(files, shuffle=shuffle)
      ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=self.interleave_cycle_length,
        block_length=self.interleave_block_length,
        num_parallel_calls=5)
      # ds = ds.prefetch(buffer_size=10*self.batch_size)
      if self.use_caching:
        ds = ds.cache()
      ds = ds.map(self._parse_and_preprocess, num_parallel_calls=10)
      if self.is_training:
        ds = ds.shuffle(self.shuffle_buffer_size).repeat()
      ds = ds.batch(self.batch_size_per_split)
      ds = ds.prefetch(buffer_size=5*self.num_splits)
      # if self.num_threads:
      #   ds = threadpool.override_threadpool(
      #     ds,
      #     threadpool.PrivateThreadPool(
      #       self.num_threads, display_name='input_pipeline_thread_pool'))
      # multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
      #     ds,
      #     self.gpu_devices,
      #     source_device=self.cpu_device)
      # tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
      #                      multi_device_iterator.initializer)
      # return multi_device_iterator
      iterator = ds.make_one_shot_iterator()
      multi_device_iterator = MultiDeviceIterator(iterator, self.gpu_devices)
      return multi_device_iterator


class MNISTReader(BaseReader):

  def __init__(self, params, batch_size, gpu_devices, cpu_device,
               is_training):
    super(MNISTReader, self).__init__(
      params, batch_size, gpu_devices, cpu_device, is_training)

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 28, 28
    self.n_train_files = 60000
    self.n_test_files = 10000
    self.n_classes = 10
    self.batch_shape = (None, 32, 32, 1)
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
    image = normalize_image(image)
    return image


class FashionMNISTReader(BaseReader):

  def __init__(self, params, batch_size, gpu_devices, cpu_device,
               is_training):
    super(FashionMNISTReader, self).__init__(
      params, batch_size, gpu_devices, cpu_device, is_training)

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 28, 28
    self.n_train_files = 60000
    self.n_test_files = 10000
    self.n_classes = 10
    self.batch_shape = (None, 32, 32, 1)
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
    image = normalize_image(image)
    return image


class CIFARReader(BaseReader):

  def __init__(self, params, batch_size, gpu_devices, cpu_device,
               is_training):
    super(CIFARReader, self).__init__(
      params, batch_size, gpu_devices, cpu_device, is_training)

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 32, 32
    self.n_train_files = 50000
    self.n_test_files = 10000
    self.n_classes = 10
    self.batch_shape = (None, 32, 32, 3)
    self.use_data_augmentation = self.params.reader['data_augmentation']
    self.use_gray_scale = self.params.reader['grayscale']
    if self.use_gray_scale:
      self.batch_shape = (None, 32, 32, 1)

  def _data_augmentation(self, image):
    image = tf.image.resize_with_crop_or_pad(
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
    image = normalize_image(image)
    return image


class CIFAR10Reader(CIFARReader):

  def __init__(self, params, batch_size, gpu_devices, cpu_device,
               is_training):
    super(CIFAR10Reader, self).__init__(
      params, batch_size, gpu_devices, cpu_device, is_training)
    self.files = self._get_tfrecords('cifar10')


class CIFAR100Reader(CIFARReader):

  def __init__(self, params, batch_size, gpu_devices, cpu_device,
               is_training):
    super(CIFAR100Reader, self).__init__(
      params, batch_size, gpu_devices, cpu_device, is_training)
    self.files = self._get_tfrecords('cifar100')



class YT8MAggregatedFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """
  def __init__(self, params, batch_size, gpu_devices, cpu_device,
               is_training):
    super(YT8MAggregatedFeatureReader, self).__init__(
      params, batch_size, gpu_devices, cpu_device, is_training)

    self.batch_size = batch_size
    self.is_training = is_training
    self.n_train_files = 3888919
    self.n_test_files = 1112356 # public
    # self.n_test_files = 1133323 # private
    self.n_classes = 3862
    self.batch_shape = (None, 1152)
    self.feature_sizes = [1024, 128]
    self.feature_names = ["mean_rgb", "mean_audio"]
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
  def __init__(self, params, batch_size, gpu_devices, cpu_device,
               is_training):
    super(YT8MFrameFeatureReader, self).__init__(
      params, batch_size, gpu_devices, cpu_device, is_training)

    self.batch_size = batch_size
    self.is_training = is_training
    self.n_train_files = 3888919
    self.n_test_files = 1112356 # public
    self.n_classes = 3862
    self.batch_shape = (None, 300, 1152)

    self.feature_sizes = [1024, 128]
    self.feature_names = ["rgb", "audio"]
    self.max_frames = 300

    self.max_quantized_value = 2
    self.min_quantized_value = -2

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

  def __init__(self, params, batch_size, gpu_devices, cpu_device,
               is_training):
    super(IMAGENETReader, self).__init__(
      params, batch_size, gpu_devices, cpu_device, is_training)

    # Provide square images of this size.
    self.image_size = self.params.reader['image_size']

    self.height, self.width = self.image_size, self.image_size
    self.n_train_files = 1281167
    self.n_test_files = 50000
    self.n_classes = 1001
    self.batch_shape = (None, self.height, self.height, 3)
    self.augmentation_strategy = self.params.reader['augmentation_strategy']
    self.use_bfloat16 = self.params.use_fp16

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

  def distorted_bounding_box_crop(self,
                                  image_bytes,
                                  bbox,
                                  min_object_covered=0.1,
                                  aspect_ratio_range=(0.75, 1.33),
                                  area_range=(0.05, 1.0),
                                  max_attempts=100,
                                  scope=None):
    """Generates cropped_image using one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
      image_bytes: `Tensor` of binary image data.
      bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
          where each coordinate is [0, 1) and the coordinates are arranged
          as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
          image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding
          box supplied.
      aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
      scope: Optional `str` for name scope.
    Returns:
      cropped image `Tensor`
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image_bytes, bbox]):
      shape = tf.image.extract_jpeg_shape(image_bytes)
      sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
          shape,
          bounding_boxes=bbox,
          min_object_covered=min_object_covered,
          aspect_ratio_range=aspect_ratio_range,
          area_range=area_range,
          max_attempts=max_attempts,
          use_image_if_no_bounding_boxes=True)
      bbox_begin, bbox_size, _ = sample_distorted_bounding_box
  
      # Crop the image to the specified bounding box.
      offset_y, offset_x, _ = tf.unstack(bbox_begin)
      target_height, target_width, _ = tf.unstack(bbox_size)
      crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
      image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  
      return image
  
  def _at_least_x_are_equal(self, a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)
  
  def _decode_and_random_crop(self, image_bytes, image_size):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = self.distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0),
        max_attempts=10,
        scope=None)
    original_shape = tf.image.extract_jpeg_shape(image_bytes)
    bad = self._at_least_x_are_equal(original_shape, tf.shape(image), 3)
  
    image = tf.cond(
        bad,
        lambda: self._decode_and_center_crop(image_bytes, image_size),
        lambda: tf.image.resize_bicubic([image],  # pylint: disable=g-long-lambda
                                        [image_size, image_size])[0])
  
    return image
  
  def _decode_and_center_crop(self, image_bytes, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]
  
    padded_center_crop_size = tf.cast(
      ((image_size / (image_size + 32)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)
  
    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
    return image
  
  def _flip(self, image):
    """Random horizontal image flip."""
    image = tf.image.random_flip_left_right(image)
    return image
  
  def preprocess_for_train(self, image_bytes):
    """Preprocesses the given image for training.
    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      image_size: image size.
    Returns:
      A preprocessed image `Tensor`.
    """
    image_size = self.image_size
    image = self._decode_and_random_crop(image_bytes, image_size)
    image = self._flip(image)
    image = tf.reshape(image, [image_size, image_size, 3])
  
    image = tf.image.convert_image_dtype(
        image, dtype=tf.bfloat16 if self.use_bfloat16 else tf.float32)
  
    if self.augmentation_strategy == 'autoaugment':
      logging.info('Apply AutoAugment policy')
      input_image_type = image.dtype
      image = tf.clip_by_value(image, 0.0, 255.0)
      image = tf.cast(image, dtype=tf.uint8)
      image = autoaugment.distort_image_with_autoaugment(image, 'v0')
      image = tf.cast(image, dtype=input_image_type)
    return image
  
  def preprocess_for_eval(self, image_bytes):
    """Preprocesses the given image for evaluation.
    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      use_bfloat16: `bool` for whether to use bfloat16.
    Returns:
      A preprocessed image `Tensor`.
    """
    image_size = self.image_size
    image = self._decode_and_center_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.image.convert_image_dtype(
        image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
    return image

  def _image_preprocessing(self, image_buffer, bbox):
    """Decode and preprocess one image for evaluation or training.
    Args:
      image: image as numpy array
    Returns:
      3D Tensor containing an appropriately scaled image
    """
    if self.is_training:
      image = self.preprocess_for_train(image_buffer)
    else:
      image = self.preprocess_for_eval(image_buffer)
    return image




class ImagenetScatteringReader(BaseReader):

  def __init__(self, image_size, j,
               params, batch_size, gpu_devices, cpu_device,
               is_training):
    super(ImagenetScatteringReader, self).__init__(
      params, batch_size, gpu_devices, cpu_device, is_training)

    channel = int(1 + j * 8 + (8**2  * j * (j - 1) / 2))
    output_size  = int(image_size / 2**j)
    self.reshape = (3, channel, output_size, output_size)

    self.n_train_files = 1281167
    self.n_test_files = 50000
    self.n_classes = 1001
    self.batch_shape = (None, self.height, self.height, 1)

    self.files = self._get_tfrecords('imagenet_{}_scattering_j{}'.format(
      image_size, j))

  def _image_preprocessing(self, image_buffer):
    """Decode and preprocess one image for evaluation or training.
    Args:
      image_buffer: JPEG encoded string Tensor
    Returns:
      Tensor containing an appropriately scaled image
    """
    image = tf.decode_raw(image_buffer, tf.float32)
    image = tf.reshape(image, self.reshape)
    return image


def create_imagenet_scattering(*args, **kwargs):
  image_size = kwargs['image_size']
  j = kwargs['j']
  del kwargs['image_size'], kwargs['j']
  return ImagenetScatteringReader(image_size, j, *args, **kwargs)


readers_config = {
  'mnist': MNISTReader,
  'cifar10': CIFAR10Reader,
  'cifar100': CIFAR100Reader,
  'fashion_mnist': FashionMNISTReader,
  'youtube_agg': YT8MAggregatedFeatureReader,
  'youtube_frame': YT8MFrameFeatureReader,
  'imagenet': IMAGENETReader,
  'imagenet_296_scattering_j1':
    partial(create_imagenet_scattering, image_size=296, j=1),
  'imagenet_296_scattering_j2':
    partial(create_imagenet_scattering, image_size=296, j=2),
  'imagenet_296_scattering_j3':
    partial(create_imagenet_scattering, image_size=296, j=3),
  'imagenet_224_scattering_j1':
    partial(create_imagenet_scattering, image_size=224, j=1),
  'imagenet_224_scattering_j2':
    partial(create_imagenet_scattering, image_size=224, j=2),
  'imagenet_224_scattering_j3':
    partial(create_imagenet_scattering, image_size=224, j=3),
  'imagenet_224_scattering_j4':
    partial(create_imagenet_scattering, image_size=224, j=4),
}


