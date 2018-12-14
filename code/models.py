
import re
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

import layers
import utils
from resnet import Resnet

from config import hparams as FLAGS



class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):
    raise NotImplementedError()



################################################################################
#####                             MNIST MODEL                              #####
################################################################################

class MnistModelDense(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.dense
    assert config['n_layers'] == len(config['hidden'])

    activation = tf.layers.flatten(model_input)

    for i in range(config['n_layers']):
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][i] or feature_size
      initializer = tf.random_normal_initializer(stddev=1/np.sqrt(feature_size))
      activation = tf.layers.dense(activation, num_hidden, activation=tf.nn.relu,
                                   kernel_initializer=initializer)
      tf.summary.histogram('Mnist/Dense/Layer{}'.format(i), activation)

    # classification layer
    feature_size = activation.get_shape().as_list()[-1]
    initializer = tf.random_normal_initializer(stddev=1/np.sqrt(feature_size))
    activation = tf.layers.dense(activation, n_classes, activation=None,
                                 kernel_initializer=initializer)
    return activation


class MnistModelGivens(BaseModel):

  def _givens_layers(self, activation, n_givens, shape_in, shape_out=None):
    for i in range(n_givens):
      givens_layer = layers.GivensLayer(shape_in, shape_out=None)
      activation = givens_layer.matmul(activation)
    if shape_out is not None:
      activation = activation[..., :shape_out]
    return activation

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.givens
    assert config['n_layers'] == len(config['hidden'])

    activation = tf.layers.flatten(model_input)

    for i in range(config['n_layers']):
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][i] or feature_size
      activation = self._givens_layers(activation, config['n_givens'],
        feature_size, num_hidden)
      activation = tf.nn.relu(activation)
      tf.summary.histogram('Mnist/Givens/Layer{}'.format(i), activation)

    # classification layer
    activation = self._givens_layers(activation, config['n_givens'],
      feature_size, n_classes)

    return activation

class MnistModelGivens_v2(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.givens
    assert config['n_layers'] == len(config['hidden'])

    activation = tf.layers.flatten(model_input)

    for i in range(config['n_layers']):
      with tf.variable_scope('givens{}'.format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config['hidden'][i] or feature_size
        cls_layer = layers.GivensLayer_v2(feature_size, num_hidden,
                                       config['n_givens'])
        activation = cls_layer.matmul(activation)
        activation = tf.nn.tanh(activation)
        tf.summary.histogram('Mnist/Givens/Layer{}'.format(i), activation)

    # classification layer
    cls_layer = layers.GivensLayer_v2(feature_size, n_classes,
                                     config['n_givens'])
    activation = cls_layer.matmul(activation)

    return activation


class MnistModelCirculant(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.circulant
    assert config["n_layers"] == len(config["hidden"])

    activation = tf.layers.flatten(model_input)
    for i in range(config["n_layers"]):
      with tf.variable_scope("circulant{}".format(i)):
        feature_size = activation.get_shape().as_list()[-1]
        num_hidden = config["hidden"][i] or feature_size
        cls_layer = layers.CirculantLayer(feature_size, num_hidden,
                                          use_diag=config["use_diag"],
                                          use_bias=config["use_bias"])
        activation = cls_layer.matmul(activation)
        activation = tf.nn.relu(activation)

    # classification layer
    cls_layer = layers.CirculantLayer(feature_size, n_classes)
    activation = cls_layer.matmul(activation)
    return activation


class MnistModelToeplitz(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    activation = tf.layers.flatten(model_input)
    feature_size = activation.get_shape().as_list()[-1]

    layer1 = layers.ToeplitzLayer(feature_size, feature_size)
    activation = layer1.matmul(activation)
    activation = tf.nn.relu(activation)

    layer2 = layers.ToeplitzLayer(feature_size, n_classes)
    activation = layer2.matmul(activation)

    return activation


################################################################################
#####                             CIFAR MODEL                              #####
################################################################################


class Cifar10BaseModel:

  def _activation_summary(self, x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('tower_[0-9]*/', '', x.op.name)
    tf.summary.histogram('{}/activations'.format(tensor_name), x)
    tf.summary.scalar('{}/sparsity'.format(tensor_name), tf.nn.zero_fraction(x))

  def convolutional_layers(self, images):
    """Build the CIFAR-10 model convolutional layers.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      activation
    """

    # conv1
    with tf.variable_scope('conv1') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=5e-2)
      biases_initializer = tf.constant_initializer(0.1)
      kernel = tf.get_variable('weights', (5, 5, 3, 64),
                               initializer=kernel_initializer)
      biases = tf.get_variable('biases', [64], initializer=biases_initializer)

      activation = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      activation = tf.nn.bias_add(activation, biases)
      activation = tf.nn.relu(activation, name=scope.name)
      self._activation_summary(activation)

    # pool1
    activation = tf.nn.max_pool(activation,
      ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    # norm1
    activation = tf.nn.lrn(activation,
      4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=5e-2)
      biases_initializer = tf.constant_initializer(0.1)
      kernel = tf.get_variable('weights', (5, 5, 64, 64),
                               initializer=kernel_initializer)
      biases = tf.get_variable('biases', [64], initializer=biases_initializer)

      activation = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
      activation = tf.nn.bias_add(activation, biases)
      activation = tf.nn.relu(activation, name=scope.name)
      self._activation_summary(activation)

    # norm2
    activation = tf.nn.lrn(activation,
      4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2
    activation = tf.nn.max_pool(activation,
      ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    return tf.layers.flatten(activation)


class Cifar10ModelDense(BaseModel, Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.dense
    activation = self.convolutional_layers(model_input)

    with tf.variable_scope('dense1') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=1/np.sqrt(384))
      activation = tf.layers.dense(activation, config['hidden'][0], use_bias=True,
        kernel_initializer=kernel_initializer, activation=tf.nn.relu)
      self._activation_summary(activation)

    with tf.variable_scope('dense2') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=1/np.sqrt(192))
      activation = tf.layers.dense(activation, config['hidden'][1], use_bias=True,
        kernel_initializer=kernel_initializer, activation=tf.nn.relu)
      self._activation_summary(activation)

    with tf.variable_scope('dense3') as scope:
      kernel_initializer = tf.random_normal_initializer(stddev=1/np.sqrt(10))
      activation = tf.layers.dense(activation, n_classes, use_bias=True,
        kernel_initializer=kernel_initializer, activation=None)
      self._activation_summary(activation)

    return activation


class Cifar10ModelGivens(BaseModel, Cifar10BaseModel):

  def _givens_layers(self, activation, n_givens, shape_in, shape_out=None):
    for i in range(n_givens):
      givens_layer = layers.GivensLayer(shape_in, shape_out=None)
      activation = givens_layer.matmul(activation)
    if shape_out is not None:
      activation = activation[..., :shape_out]
    return activation

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.givens

    activation = self.convolutional_layers(model_input)

    assert len(config['hidden']) == 2

    with tf.variable_scope('givens1') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][0] or feature_size
      activation = self._givens_layers(activation, config["n_givens"],
        feature_size, num_hidden)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('givens2') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][1] or feature_size
      activation = self._givens_layers(activation, config["n_givens"],
        feature_size, num_hidden)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('givens3') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      activation = self._givens_layers(activation, config["n_givens"],
        feature_size, n_classes)
      self._activation_summary(activation)

    return activation


class Cifar10ModelCirculant(BaseModel, Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.circulant
    # activation = self.convolutional_layers(model_input)
    activation = tf.layers.flatten(model_input)

    with tf.variable_scope('circulant1') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][0] or feature_size
      layer1 = layers.CirculantLayer(feature_size, num_hidden)
      activation = layer1.matmul(activation)
      bias = tf.get_variable('bias1', shape=(num_hidden, ),
                             initializer=tf.constant_initializer(0.1))
      activation = activation + bias
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('circulant2') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = config['hidden'][1] or feature_size
      layer2 = layers.CirculantLayer(feature_size, num_hidden)
      activation = layer2.matmul(activation)
      bias = tf.get_variable('bias2', shape=(num_hidden, ),
                             initializer=tf.constant_initializer(0.1))
      activation = activation + bias
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('circulant3') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer3 = layers.CirculantLayer(feature_size, n_classes)
      activation = layer3.matmul(activation)
      bias = tf.get_variable('bias3', shape=(n_classes, ),
                            initializer=tf.constant_initializer(0.1))
      activation = activation + bias
      self._activation_summary(activation)

    return activation



class Cifar10ModelToeplitz(BaseModel, Cifar10BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    activation = self.convolutional_layers(model_input)

    with tf.variable_scope('toeplitz1') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer1 = layers.ToeplitzLayer(feature_size, 384)
      activation = layer1.matmul(activation)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('toeplitz2') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer2 = layers.ToeplitzLayer(feature_size, 192)
      activation = layer2.matmul(activation)
      activation = tf.nn.relu(activation)
      self._activation_summary(activation)

    with tf.variable_scope('toeplitz3') as scope:
      feature_size = activation.get_shape().as_list()[-1]
      layer3 = layers.ToeplitzLayer(feature_size, 10)
      activation = layer3.matmul(activation)
      self._activation_summary(activation)

    return activation




################################################################################
#####                            RESNET MODEL                              #####
################################################################################

class ResnetModel(BaseModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    config = FLAGS.resnet

    output = Resnet(
      resnet_size=config['resnet_size'],
      bottleneck=config['bottleneck'],
      num_classes=n_classes,
      num_filters=config['num_filters'],
      kernel_size=config['kernel_size'],
      conv_stride=config['conv_stride'],
      first_pool_size=config['first_pool_size'],
      first_pool_stride=config['first_pool_stride'],
      second_pool_size=config['second_pool_size'],
      second_pool_stride=config['second_pool_stride'],
      block_sizes=config['block_sizes'],
      block_strides=config['block_strides'],
      final_size=config['final_size'],
      version=config['version'],
      data_format=config['data_format']
    ).foward(model_input, is_training)
    return output


################################################################################
#####                            RANDOM MODEL                              #####
################################################################################


class RandomModel:

  def _noise_layer(self, x, sensitivity_norm, sensitivity_control_scheme):
    """Pixeldp noise layer."""
    # This is a factor applied to the noise layer,
    # used to rampup the noise at the beginning of training.
    dp_mult = self._dp_mult(sensitivity_norm)
    if sensitivity_control_scheme == 'optimize':
      sensitivity = tf.reduce_prod(self._sensitivities)
    elif sensitivity_control_scheme == 'bound':
      sensitivity = 1

    if self.is_training:
      # global_step = tf.cast(tf.train.get_global_step(), tf.float32)
      # noise_scale = 1.0 - tf.minimum(0.99975**global_step, 0.9)
      noise_scale = tf.constant(1.0)
    else:
      if FLAGS.noise_in_eval:
        noise_scale = tf.constant(1.0)
      else:
        noise_scale = tf.constant(0.0)

    final_noise_scale = noise_scale * dp_mult * sensitivity
    if sensitivity_norm == 'l1':
      laplace_shape = tf.shape(x)
      loc = tf.zeros(laplace_shape, dtype=tf.float32)
      scale = tf.ones(laplace_shape,  dtype=tf.float32)
      noise = tf.distributions.Laplace(loc, scale).sample()
      noise = final_noise_scale * noise

    elif sensitivity_norm == 'l2':
      noise = tf.random_normal(tf.shape(x), mean=0, stddev=1)
      noise = final_noise_scale * noise

    return x + noise

  def _dp_mult(self, sensitivity_norm, output_dim=None):
    dp_eps = self.config['dp_epsilon']
    dp_del = self.config['dp_delta']
    if sensitivity_norm == 'l2':
      # Use the Gaussian mechanism
      return self.config['attack_norm_bound'] * \
           np.sqrt(2 * np.log(1.25 / dp_del)) / dp_eps
    elif sensitivity_norm == 'l1':
      # Use the Laplace mechanism
      return self.config['attack_norm_bound'] / dp_eps
    else:
      return 0


  def _conv(self, name, x, filter_size, in_filters, out_filters,
            strides, position=None):
    """Convolution, with support for sensitivity bounds when they are
    pre-noise."""

    assert(strides[1] == strides[2])
    stride = strides[1]

    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      shape = [filter_size, filter_size, in_filters, out_filters]
      kernel = tf.get_variable('kernel{}'.format(position), shape,
        initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)),
        regularizer=self.regularizer)

      if position == None or position > self.config['noise_after_n_layers']:
        # Post noise: no sensitivity control.
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')

      sensitivity_control_scheme = self.config['sensitivity_control_scheme']
      layer_sensivity_bound = self.layer_sensitivity_bounds[position-1]

      if layer_sensivity_bound == 'l2_l2':
        # Parseval projection, see: https://arxiv.org/abs/1704.08847
        tf.add_to_collection("parseval_convs", kernel)
        sensitivity_rescaling = np.ceil(filter_size / stride)
        k = kernel / sensitivity_rescaling

        if sensitivity_control_scheme == 'optimize':
          raise ValueError("Cannot optimize sensitivity for l2_l2.")
        elif sensitivity_control_scheme == 'bound':
          # Compute the sensitivity and keep it.
          # Use kernel as we compensate to the reshapre by using k in
          # the conv2d.
          shape = kernel.get_shape().as_list()
          w_t = tf.reshape(kernel, [-1, shape[-1]])
          w = tf.transpose(w_t)
          sing_vals = tf.svd(w, compute_uv=False)
          self._sensitivities.append(tf.reduce_max(sing_vals))
          return tf.nn.conv2d(x, k, strides, padding='SAME')

      elif layer_sensivity_bound == 'l1_l2':
        if sensitivity_control_scheme == 'optimize':
          k = kernel
        elif sensitivity_control_scheme == 'bound':
          # Sensitivity 1 by L2 normalization
          k = tf.nn.l2_normalize(kernel, dim=[0, 1, 3])

        # Compute the sensitivity
        sqr_sum  = tf.reduce_sum(tf.square(k), [0, 1, 3], keep_dims=True)
        l2_norms = tf.sqrt(sqr_sum)
        self._sensitivities.append(tf.reduce_max(l2_norms))
        return tf.nn.conv2d(x, k, strides, padding='SAME')

      elif layer_sensivity_bound == 'l1_l1':
        if sensitivity_control_scheme == 'optimize':
          k = kernel
        elif sensitivity_control_scheme == 'bound':
          # Sensitivity 1 by L1 normalization
          k = utils.l1_normalize(kernel, dim=[0, 1, 3])

        # Compute the sensitivity
        l1_norms = tf.reduce_sum(tf.abs(k), [0, 1, 3], keep_dims=True)
        self._sensitivities.append(tf.reduce_max(l1_norms))
        return tf.nn.conv2d(x, k, strides, padding='SAME')

      else:
        raise ValueError("Pre-noise with unsupported  sensitivity.")

  def _stride_arr(self, stride):
      """Map a stride scalar to the stride array for tf.nn.conv2d."""
      return [1, stride, stride, 1]


class MnistRandomModel(RandomModel):

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):

    self.config = config = FLAGS.random_model
    self.n_classes = n_classes
    self.is_training = is_training

    assert(config['noise_after_n_layers'] <= 2)

    # To allow specifying only one config['layer_sensitivity_bounds'] that is
    # used for every layer.
    # TODO: assert that the bounds make sense w.r.t each other and 
    # start/end norms.
    self.layer_sensitivity_bounds = config['layer_sensitivity_bounds']
    noise_after_n_layers = config['noise_after_n_layers']
    if len(self.layer_sensitivity_bounds) == 1 and noise_after_n_layers > 1:
      self.layer_sensitivity_bounds = \
          self.layer_sensitivity_bounds * noise_after_n_layers

    # Book keeping for the noise layer
    self._sensitivities = [1]
    sensitivity_norm = config['sensitivity_norm']
    sensitivity_control_scheme = config['sensitivity_control_scheme']

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      self.regularizer = None
    else:
      self.regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    x = model_input

    if config['noise_after_n_layers'] == 0:
      x = self._noise_layer(x, sensitivity_norm, sensitivity_control_scheme)

    with tf.variable_scope('conv1'):
      filter_size = 5
      in_filters  = x.get_shape()[-1]
      out_filters = 32
      strides = [1, 2, 2, 1]
      x = self._conv("conv1", x, filter_size, in_filters, out_filters,
                     strides, position=1)
      if config['noise_after_n_layers'] == 1:
        x = self._noise_layer(x, sensitivity_norm, sensitivity_control_scheme)
      x = tf.nn.relu(x)

    with tf.variable_scope('conv2'):
      strides = [1, 2, 2, 1]
      x = self._conv("conv2", x, 5, out_filters, 64, strides, position=2)
      if config['noise_after_n_layers'] == 2:
        self._noise_layer(x, sensitivity_norm, sensitivity_control_scheme)
      x = tf.nn.relu(x)

    x = tf.layers.flatten(x)

    with tf.variable_scope('dense1') as scope:
      feature_size = x.get_shape().as_list()[-1]
      stddev = 1/np.sqrt(feature_size)
      kernel_initializer = tf.random_normal_initializer(stddev=stddev)
      x = tf.layers.dense(x, 1024, use_bias=True,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=self.regularizer,
        bias_regularizer=self.regularizer,
        activation=tf.nn.relu)

    with tf.variable_scope('logits') as scope:
      feature_size = x.get_shape().as_list()[-1]
      stddev = 1/np.sqrt(feature_size)
      kernel_initializer = tf.random_normal_initializer(stddev=stddev)
      logits = tf.layers.dense(x, n_classes, use_bias=True,
         kernel_initializer=kernel_initializer,
         kernel_regularizer=self.regularizer,
         bias_regularizer=self.regularizer,
         activation=None)

    return logits



class Cifar10RandomModel(RandomModel):
  """ResNet model."""

  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                    activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x, self.config['leakyness'])
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x, self.config['leakyness'])

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2'):
      x = tf.layers.batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.config['leakyness'])
      x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      x = tf.layers.batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.config['leakyness'])
      x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x
    return x

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

  def _residual(self, x, in_filter, out_filter, stride,
           activate_before_residual=False):
   """Residual unit with 2 sub layers."""
   if activate_before_residual:
     with tf.variable_scope('shared_activation'):
       x = tf.layers.batch_normalization(x)
       x = tf.nn.leaky_relu(x, self.config['leakyness'])
       orig_x = x
   else:
     with tf.variable_scope('residual_only_activation'):
       orig_x = x
       x = tf.layers.batch_normalization(x)
       x = tf.nn.leaky_relu(x, self.config['leakyness'])

   with tf.variable_scope('sub1'):
     x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

   with tf.variable_scope('sub2'):
     x = tf.layers.batch_normalization(x)
     x = tf.nn.leaky_relu(x, self.config['leakyness'])
     x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

   with tf.variable_scope('sub_add'):
     if in_filter != out_filter:
       orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
       orig_x = tf.pad(
         orig_x, [[0, 0], [0, 0], [0, 0],
               [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
     x += orig_x
   return x


  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):
    """Build the core model within the graph."""

    self.config = config = {**FLAGS.random_model, **FLAGS.cifar_random_model}
    self.n_classes = n_classes
    self.is_training = is_training

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      self.regularizer = None
    else:
      self.regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    logging.info("model shape".format(model_input.get_shape()))

    # Book keeping for the noise layer
    self._sensitivities = [1]
    sensitivity_norm = config['sensitivity_norm']
    sensitivity_control_scheme = config['sensitivity_control_scheme']

    # To allow specifying only one config['layer_sensitivity_bounds'] that is
    # used for every layer.
    # TODO: assert that the bounds make sense w.r.t each other and 
    # start/end norms.
    self.layer_sensitivity_bounds = config['layer_sensitivity_bounds']
    noise_after_n_layers = config['noise_after_n_layers']
    if len(self.layer_sensitivity_bounds) == 1 and noise_after_n_layers > 1:
      self.layer_sensitivity_bounds = \
          self.layer_sensitivity_bounds * noise_after_n_layers

    x = model_input

    if config['noise_after_n_layers'] == 0:
      x = self._noise_layer(x, sensitivity_norm, sensitivity_control_scheme)

    with tf.variable_scope('init'):
      filter_size = 3
      in_filters  = x.get_shape()[-1]
      out_filters = 16
      strides = [1, 1, 1, 1]
      x = self._conv("init_conv", x, filter_size, in_filters, out_filters,
            strides, position=1)
      if config['noise_after_n_layers'] == 1:
        x = self._noise_layer(x, sensitivity_norm, sensitivity_control_scheme)
      x = tf.nn.leaky_relu(x, config['leakyness'])

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if config['use_bottleneck']:
      res_func = self._bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      res_func = self._residual
      filters = [16, 16, 32, 64]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      # filters = [out_filters, 160, 320, 640]
      # Update hps.num_residual_units to 4

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
        activate_before_residual[0])
    for i in range(1, config['num_residual_units']):
      with tf.variable_scope('unit_1_{}'.format(i)):
         x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
               activate_before_residual[1])
    for i in range(1, config['num_residual_units']):
      with tf.variable_scope('unit_2_{}'.format(i)):
         x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
               activate_before_residual[2])
    for i in range(1, config['num_residual_units']):
      with tf.variable_scope('unit_3_{}'.format(i)):
         x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = tf.layers.batch_normalization(x)
      x = tf.nn.leaky_relu(x, config['leakyness'])
      x = self._global_avg_pool(x)

    logging.info("after conv {}".format(x.get_shape()))

    with tf.variable_scope('logits') as scope:
      feature_size = x.get_shape().as_list()[-1]
      stddev = 1/np.sqrt(feature_size)
      kernel_initializer = tf.random_normal_initializer(stddev=stddev)
      logits = tf.layers.dense(x, n_classes, use_bias=True,
         kernel_initializer=kernel_initializer,
         kernel_regularizer=self.regularizer,
         bias_regularizer=self.regularizer,
         activation=None)

    return logits



