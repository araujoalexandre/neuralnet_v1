"""The CarliniWagnerL2 attack
"""
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf_dtype = tf.float32


class CarliniWagnerL2:
  """
  This attack was originally proposed by Carlini and Wagner. It is an
  iterative attack that finds adversarial examples on many defenses that
  are robust to other attacks.
  Paper link: https://arxiv.org/abs/1608.04644
  At a high level, this attack is an iterative attack using Adam and
  a specially-chosen loss function to find adversarial examples with
  lower distortion than other attacks. This comes at the cost of speed,
  as this attack is often much slower than others.
  """

  def __init__(self,
               y_target=None,
               batch_size=1,
               confidence=0,
               learning_rate=5e-3,
               binary_search_steps=5,
               max_iterations=1000,
               abort_early=True,
               initial_const=1e-2,
               clip_min=-1,
               clip_max=+1,
               sample=1):
    """
    :param y_target: (optional) A tensor with the target labels for a
              targeted attack.
    :param confidence: Confidence of adversarial examples: higher produces
                       examples with larger l2 distortion, but more
                       strongly classified as adversarial.
    :param batch_size: Number of attacks to run simultaneously.
    :param learning_rate: The learning rate for the attack algorithm.
                          Smaller values produce better results but are
                          slower to converge.
    :param binary_search_steps: The number of times we perform binary
                                search to find the optimal tradeoff-
                                constant between norm of the purturbation
                                and confidence of the classification.
    :param max_iterations: The maximum number of iterations. Setting this
                           to a larger value will produce lower distortion
                           results. Using only a few iterations requires
                           a larger learning rate, and will produce larger
                           distortion results.
    :param abort_early: If true, allows early aborts if gradient descent
                        is unable to make progress (i.e., gets stuck in
                        a local minimum).
    :param initial_const: The initial tradeoff-constant to use to tune the
                          relative importance of size of the perturbation
                          and confidence of classification.
                          If binary_search_steps is large, the initial
                          constant is not important. A smaller value of
                          this constant gives lower distortion results.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """
    self.y_target = y_target
    # self.batch_size = batch_size
    self.confidence = confidence
    self.learning_rate = learning_rate
    self.binary_search_steps = binary_search_steps
    self.max_iterations = max_iterations
    self.abort_early = abort_early
    self.initial_const = initial_const
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sample = sample

    logging.info("max_iterations: {}".format(max_iterations))
    # logging.info("batch_size: {}".format(batch_size))

  def get_name(self):
    return 'Carlini_{}_{}_{}'.format(
      self.max_iterations, self.binary_search_steps, self.sample)

  def generate(self, x, fn_logits, y=None):
    self.fn_logits = fn_logits
    # wrap attack function in py_func
    def cw_wrap(x_val):
      return self.attack(x_val)
    adv = tfe.py_func(cw_wrap, [x], tf_dtype)
    adv.set_shape(x.get_shape())
    return adv

  def attack(self, imgs):
    """
    Return a tensor that constructs adversarial examples for the given
    input. Generate uses tf.py_func in order to operate over tensors.
    :param x: A tensor with the inputs.
    :param kwargs: See `parse_params`
    """
    imgs = tf.cast(imgs, tf.float32)
    preds = self.fn_logits(imgs)
    preds_max = tf.reduce_max(preds, 1, keepdims=True)
    original_predictions = tf.to_float(tf.equal(preds, preds_max))
    labs = tf.stop_gradient(original_predictions)
    repeat = self.binary_search_steps >= 10
    shape = tf.shape(imgs)

    # # the variable we're going to optimize over
    # modifier = tfe.Variable(tf.zeros(shape, dtype=tf_dtype))

    def compute_newimage(imgs, modifier):
      # the resulting instance, tanh'd to keep bounded from clip_min
      # to clip_max
      newimg = (tf.tanh(modifier + imgs) + 1) / 2
      newimg = newimg * (self.clip_max - self.clip_min) + self.clip_min
      return newimg

    def get_l2dist(imgs, newimg):
      # distance to the input data
      other = (tf.tanh(imgs) + 1) / 2 * (self.clip_max - self.clip_min) + self.clip_min
      sum_axis = list(range(1, len(shape.numpy())))
      l2dist = tf.reduce_sum(tf.square(newimg - other), sum_axis)
      return l2dist

    def loss(timg, tlab, const, modifier):
      newimg = compute_newimage(timg, modifier)
      # prediction BEFORE-SOFTMAX of the model
      if self.sample <= 1:
        output = self.fn_logits(newimg)
      else:
        logging.info(
          "Monte Carlo (MC) on attacks, sample: {}".format(self.sample))
        for i in range(self.sample):
          logits = self.fn_logits(newimg)
          if i == 0:
            assert logits.op.type != 'Softmax'
          output.append(logits)
        output = tf.reduct_mean(output, 0)

      # distantce to the input data
      l2dist = get_l2dist(timg, newimg)

      # compute the probability of the label class versus the maximum other
      real_target = tf.reduce_sum((tlab) * output, 1)
      other_target = tf.reduce_max((1 - tlab) * output - tlab * 10000, 1)
      zero = tf.constant(0., dtype=tf_dtype)
      if self.y_target:
        # if targeted, optimize for making the other class most likely
        loss1 = tf.maximum(zero, other_target - real_target + self.confidence)
      else:
        # if untargeted, optimize for making this class least likely.
        loss1 = tf.maximum(zero, real_target - other_target + self.confidence)

      # sum up the losses
      loss2 = tf.reduce_sum(l2dist)
      loss1 = tf.reduce_sum(const * loss1)
      loss = loss1 + loss2
      return loss, output


    def grad(imgs, labs, const, modifier):
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(modifier)
        loss_value, logits = loss(imgs, labs, const, modifier)
        with tape.stop_recording():
          gradients = tape.gradient(loss_value, [modifier])
      return gradients, loss_value, logits


    def compare_multi(x, y):
      x_array = tf.unstack(x)
      if self.y_target:
        x_array[y] = x_array[y] - self.confidence
      else:
        x_array[y] = x_array[y] + self.confidence
      x = tf.argmax(tf.stack(x_array))
      if self.y_target:
        return x == y
      else:
        return x != y

    def compare_single(x, y):
      if self.y_target:
        return x == y
      else:
        return x != y


    # batch_size = tf.shape(imgs)[0]
    batch_size = imgs.get_shape().as_list()[0]

    # re-scale instances to be within range [0, 1]
    imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
    imgs = tf.clip_by_value(imgs, 0, 1)
    # now convert to [-1, 1]
    imgs = (imgs * 2) - 1
    # convert to tanh-space
    imgs = tf.atanh(imgs * .999999)

    # set the lower and upper bounds accordingly
    lower_bound = tfe.Variable(tf.zeros(batch_size), trainable=False)
    const = tfe.Variable(tf.ones(batch_size) * self.initial_const, trainable=False)
    upper_bound = tfe.Variable(tf.ones(batch_size) * 1e10, trainable=False)

    # placeholders for the best l2, score, and instance attack found so far
    o_bestl2 = tfe.Variable(tf.constant(1e10, shape=(batch_size, )), trainable=False)
    o_bestscore = tfe.Variable(tf.constant(-1, shape=(batch_size, )), trainable=False)
    o_bestattack = tfe.Variable(tf.identity(imgs), trainable=False)

    for outer_step in range(self.binary_search_steps):

      # completely reset adam's internal state.
      modifier = tfe.Variable(tf.zeros(shape, dtype=tf_dtype))
      optimizer = tf.train.AdamOptimizer(self.learning_rate)

      bestl2 = tfe.Variable(tf.constant(1e10, shape=(batch_size, )), trainable=False)
      bestscore = tfe.Variable(tf.constant(-1, shape=(batch_size, )), trainable=False)
      logging.info("  Binary search step %s of %s",
                    outer_step, self.binary_search_steps)

      # The last iteration (if we run many steps) repeat the search once.
      if repeat and outer_step == self.binary_search_steps - 1:
        const = upper_bound

      prev = 1e6
      for iteration in range(self.max_iterations):

        import resource, gc
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logging.info('resource {}'.format(mem))
        gc.collect()

        tf.set_random_seed(np.random.randint(0, 100))

        # perform the attack
        gradients, loss_value, scores = grad(imgs, labs, const, modifier)
        optimizer.apply_gradients(zip(gradients, [modifier]))

        nimg = compute_newimage(imgs, modifier)
        l2s = get_l2dist(imgs, nimg)

        if iteration % ((self.max_iterations // 10) or 1) == 0 and \
           logging.get_verbosity() == logging.DEBUG:
          l2_mean = tf.reduce_mean(l2s).numpy()
          logging.debug(
            "    Iteration {} of {}: loss={:.3g} l2={:.3g}".format(
              iteration, self.max_iterations, loss_value, l2_mean))

        # check if we should abort search if we're getting nowhere.
        if self.abort_early and \
           iteration % ((self.max_iterations // 10) or 1) == 0:
          if loss_value > prev * .9999:
            logging.debug("    Failed to make progress; stop early" )
            break
          prev = loss_value

        # adjust the best result found so far
        for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
          lab = tf.argmax(labs[e])
          comp = compare_multi(sc, lab)
          if l2 < bestl2[e] and comp:
            bestl2[e].assign(l2)
            bestscore[e].assign(tf.argmax(sc, output_type=tf.int32))
          if l2 < o_bestl2[e] and comp:
            o_bestl2[e].assign(l2)
            o_bestscore[e].assign(tf.argmax(sc, output_type=tf.int32))
            o_bestattack[e].assign(ii)

      # adjust the constant as needed
      for e in range(batch_size):
        if compare_single(bestscore[e], tf.argmax(labs[e])) and bestscore[e] != -1:
          # success, divide const by two
          upper_bound[e].assign(tf.minimum(upper_bound[e], const[e]))
          if upper_bound[e] < 1e9:
            const[e].assign((lower_bound[e] + upper_bound[e]) / 2)
        else:
          # failure, either multiply by 10 if no solution found yet
          #          or do binary search with the known upper bound
          lower_bound[e].assign(tf.maximum(lower_bound[e], const[e]))
          if upper_bound[e] < 1e9:
            const[e].assign((lower_bound[e] + upper_bound[e]) / 2)
          else:
            const[e].assign(const[e]*10)

      if logging.get_verbosity() == logging.DEBUG:
        success = tf.cast(tf.less(upper_bound, 1e9), tf.int32)
        logging.debug("  Successfully generated adversarial examples " +
                      "on {} of {} instances.".format(
                          tf.reduce_sum(success), batch_size))

        mask = tf.less(o_bestl2, 1e9)
        mean = tf.reduce_mean(tf.sqrt(tf.boolean_mask(o_bestl2, mask)))
        logging.debug("   Mean successful distortion: {:.4g}".format(mean.numpy()))

    # return the best solution found
    success = tf.cast(tf.less(upper_bound, 1e9), tf.int32)
    logging.info("  Successfully generated adversarial examples " +
                 "on {} of {} instances.".format(
                      tf.reduce_sum(success), batch_size))

    mask = tf.less(o_bestl2, 1e9)
    mean = tf.reduce_mean(tf.sqrt(tf.boolean_mask(o_bestl2, mask)))
    logging.info("   Mean successful distortion: {:.4g}".format(mean.numpy()))
    return o_bestattack.read_value()



