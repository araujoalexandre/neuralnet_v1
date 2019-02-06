import numpy as np
import tensorflow as tf
from cleverhans import attacks as cleverhans_attack

from config import YParams
from config import hparams as FLAGS


class Attacks:

  def __init__(self, sess, model):
    self.config = FLAGS.attack
    self.sess = sess
    attack_cls = getattr(cleverhans_attack, FLAGS.attack_method)
    self.attack = attack_cls(model, sess=sess)

    # get attack params
    self.kwargs = self.config.get(FLAGS.attack_method, {})
    for k, v in self.kwargs.items():
      if v == "default":
        del self.kwargs[k]

  def generate(self, images, labels):
    if getattr(self.attack, "generate"):
      tf_imgs = tf.convert_to_tensor(images)
      tf_labels = tf.convert_to_tensor(labels)
      images_adv = self.sess.run(
        self.attack.generate(tf_imgs, **self.kwargs))
    elif getattr(self.attack, "generate_np"):
      images_adv = self.attack.generate_np(images, **self.kwargs)

    return images_adv
































































































































































