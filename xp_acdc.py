
import os, sys
import tensorflow as tf
from tensorflow import logging
import numpy as np

logging.set_verbosity(logging.INFO)



class CirculantLayer:

  def __init__(self, shape_in, shape_out, kernel_initializer=None,
    diag_initializer=None, bias_initializer=None, regularizer=None,
    use_diag=True, use_bias=True):

    self.use_diag, self.use_bias = use_diag, use_bias
    self.shape_in, self.shape_out = shape_in, shape_out
    size = np.max([shape_in, shape_out])

    self.size = np.max([shape_in, shape_out])
    shape = (self.size, )

    if kernel_initializer is None:
      kernel_initializer = np.random.normal(0, 0.001, size=shape)
      kernel_initializer[0] = 1 + np.random.normal(0, 0.01)
      kernel_initializer = np.float32(kernel_initializer)
      self.kernel = tf.get_variable(name='kernel',
        initializer=kernel_initializer, regularizer=regularizer)
    else:
      self.kernel = tf.get_variable(name='kernel', shape=shape,
        initializer=kernel_initializer, regularizer=regularizer)

    if diag_initializer is None:
      diag_initializer = np.float32(np.random.choice([-1, 1], size=[shape_out]))

    if bias_initializer is None:
      bias_initializer = tf.constant_initializer(0.1)

    self.padding = True
    if shape_out < shape_in:
        self.padding = False

    if use_diag:
      self.diag = tf.get_variable(name='diag', # shape=[shape_out],
        initializer=diag_initializer, regularizer=regularizer)
    if use_bias:
      self.bias = tf.get_variable(name="bias", shape=[shape_out],
        initializer=bias_initializer, regularizer=regularizer)

  def matmul(self, input_data):
    padding_size = (np.abs(self.size - self.shape_in))
    paddings = ((0, 0), (padding_size, 0))
    data = tf.pad(input_data, paddings) if self.padding else input_data
    act_fft = tf.spectral.rfft(data)
    kernel_fft = tf.spectral.rfft(self.kernel)
    ret_mul = tf.multiply(act_fft, kernel_fft)
    ret = tf.spectral.irfft(ret_mul)
    ret = tf.cast(ret, tf.float32)
    ret = ret[..., :self.shape_out]
    if self.use_diag:
      ret = tf.multiply(ret, self.diag)
    if self.use_bias:
      ret = ret + self.bias
    return ret


class ACDCLayer:

  def __init__(self, shape_in, shape_out, start=False,
               regularizer=None, use_bias=True,
               sign_init=False, rand_init=False,
               normal_init=False, alpha=1):

    self.use_bias = use_bias
    self.start = start
    self.shape_in, self.shape_out = shape_in, shape_out
    size = np.max([shape_in, shape_out])

    self.size = np.max([shape_in, shape_out])
    shape = (self.size, )


    if sign_init:

      diag_initializer = np.float32(np.random.choice([-1., 1.], size=[shape_in]))
      self.diag = tf.get_variable(name='diag', # shape=[shape_in],
               initializer=diag_initializer, regularizer=regularizer)

      kernel_initializer = np.float32(np.random.choice([-1., 1.], size=[shape_in]))
      self.kernel = tf.get_variable(name='kernel', # shape=[shape_in],
               initializer=kernel_initializer, regularizer=regularizer)

    else:
      diag_initializer = tf.constant_initializer(1.)
      self.diag = tf.get_variable(name='diag', shape=[shape_in],
               initializer=diag_initializer, regularizer=regularizer)

      kernel_initializer = tf.constant_initializer(1.)
      self.kernel = tf.get_variable(name='kernel', shape=[shape_in],
               initializer=kernel_initializer, regularizer=regularizer)

    if rand_init:

      noise = tf.random_uniform(tf.shape(self.diag), minval=-0.01, maxval=0.01)
      self.diag = tf.add(self.diag, noise)

      noise = tf.random_uniform(tf.shape(self.diag), minval=-0.01, maxval=0.01)
      self.kernel = tf.add(self.kernel, noise)

    self.padding = True
    if shape_out < shape_in:
        self.padding = False

    if use_bias:
      bias_initializer = tf.constant_initializer(0.0)
      self.bias = tf.get_variable(name="bias", shape=[shape_out],
        initializer=bias_initializer, regularizer=regularizer)

  def matmul(self, input_data):
    padding_size = (np.abs(self.size - self.shape_in))
    paddings = ((0, 0), (padding_size, 0))
    data = tf.pad(input_data, paddings) if self.padding else input_data
    data = np.multiply(data, self.diag)
    act_dct = tf.spectral.dct(data, type=2)
    ret_mul = tf.multiply(act_dct, self.kernel)
    # kernel_dct = tf.spectral.dct(self.kernel, type=2)
    # ret_mul = tf.multiply(act_dct, kernel_dct)
    ret = tf.spectral.idct(ret_mul, type=2)
    ret = ret[..., :self.shape_out]
    ret = np.multiply(ret, self.diag)
    if self.use_bias:
      ret = ret + self.bias
    return ret


def build_graph_acdc(input_data, n_layers):
  """Function to define the Computational Graph"""
  activation = tf.layers.flatten(input_data)
  for i in range(n_layers):
    with tf.variable_scope("ACDC{}".format(i)):
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = feature_size
      start = True if i == 0 else False
      cls_layer = ACDCLayer(feature_size, num_hidden,
                                   start=start,
                                   use_bias=True,
                                   regularizer=None,
                                   rand_init=True,
                                   sign_init=False,
                                   normal_init=False)
      activation = cls_layer.matmul(activation)
      activation *= 0.01
  return activation

def build_graph_acdc_with_dense(input_data, n_layers):
  """Function to define the Computational Graph"""
  logging.info("ACDC with dense matrix")
  activation = tf.layers.flatten(input_data)
  with tf.variable_scope("first"):
    feature_size = activation.get_shape().as_list()[-1]
    w = tf.Variable(tf.random_normal([feature_size, feature_size],
              mean=0, stddev=1/np.sqrt(feature_size)))
    b = tf.Variable(tf.constant(0.01, shape=[feature_size]))
    activation = tf.matmul(activation, w) + b

  for i in range(n_layers):
    with tf.variable_scope("ACDC{}".format(i)):
      feature_size = activation.get_shape().as_list()[-1]
      num_hidden = feature_size
      start = True if i == 0 else False
      cls_layer = ACDCLayer(feature_size, num_hidden,
                                   start=start,
                                   use_bias=True,
                                   regularizer=None,
                                   rand_init=True,
                                   sign_init=False,
                                   normal_init=False)
      activation = cls_layer.matmul(activation)
      activation *= 0.01

  with tf.variable_scope("last"):
    feature_size = activation.get_shape().as_list()[-1]
    w = tf.Variable(tf.random_normal([feature_size, feature_size],
              mean=0, stddev=1/np.sqrt(feature_size)))
    b = tf.Variable(tf.constant(0.01, shape=[feature_size]))
    activation = tf.matmul(activation, w) + b
  return activation

def build_graph_circ(input_data, n_layers):
  """Function to define the Computational Graph"""
  alpha = 1.41
  x = input_data
  for i in range(n_layers):
    with tf.variable_scope("circulant{}".format(i)):
      feature_size = x.get_shape().as_list()[-1]
      num_hidden = feature_size
      kernel_initializer = tf.random_normal_initializer(
        stddev=alpha/np.sqrt(feature_size + num_hidden))
      bias_initializer = tf.random_normal_initializer(stddev=0.01)
      cls_layer = CirculantLayer(feature_size, num_hidden,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 use_diag=True,
                                 use_bias=True)
      x = cls_layer.matmul(x)
  return x



def build_model(n_layers, build_graph):

  # create the x_train placeholder and y_train placeholder here!
  x_placeholder = tf.placeholder(tf.float32, shape=(None, 32))
  y_placeholder = tf.placeholder(tf.int32, shape=(None, 32))

  predictions = build_graph(x_placeholder, n_layers)

  # Define the loss: use sparse_softmax_cross_entropy
  loss = tf.losses.mean_squared_error(labels=y_placeholder, predictions=predictions)

  # Define an opimizer with tf.train
  optimizer = tf.train.GradientDescentOptimizer(0.01)

  # define the training op: when executed, 
  # this op will execute the backpropagation algorithm
  train_op = optimizer.minimize(loss)
  return train_op, loss, x_placeholder, y_placeholder

def main():

  type_, n_layers = sys.argv[1], int(sys.argv[2])

  np.random.seed(1234)
  X = np.random.uniform(size=(10000, 32))
  W = np.random.uniform(size=(32, 32))
  eps = np.random.normal(0, 10e-1, size=(10000, 32))
  Y = X @ W + eps
  (x_train, y_train), (x_test, y_test) = (X, Y), (X, Y)

  if type_ == "circ":
    train_op, loss, x_placeholder, y_placeholder = \
        build_model(n_layers, build_graph_circ)
  elif type_ == "acdc":
    train_op, loss, x_placeholder, y_placeholder = \
        build_model(n_layers, build_graph_acdc)
  elif type_ == "acdc_dense":
    train_op, loss, x_placeholder, y_placeholder = \
        build_model(n_layers, build_graph_acdc_with_dense)

  batch_size = 100
  n_batch = x_train.shape[0] / batch_size

  x_train_splited = np.array_split(x_train, n_batch)
  y_train_splited = np.array_split(y_train, n_batch)

  iteration = 0
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
      for step, (x_batch, y_batch) in enumerate(zip(x_train_splited, y_train_splited), 1):
        data_to_feed = {
            x_placeholder: x_batch,
            y_placeholder: y_batch
        }
        _, loss_ = sess.run([train_op, loss], feed_dict=data_to_feed)
        if iteration % 10 == 0:
          logging.info('{}\t{:.4f}'.format(iteration, loss_))
        iteration += 1
        if iteration >= 5000:
          sys.exit(0)

if __name__ == '__main__':
  main()

