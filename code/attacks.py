
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
    self.generate = getattr(self, FLAGS.attack_method)

  def CarliniWagnerL2(self, images):
    params = self.config['CarliniWagnerL2']
    cw_params = {
          'binary_search_steps': params['binary_search_steps'],
          'max_iterations': params['max_iterations'],
          'learning_rate': params['learning_rate'],
          'batch_size': images.shape[0],
          'initial_const': params['initial_const']
    }
    images_adv = self.attack.generate_np(images, **cw_params)
    return images_adv

  def FastGradientMethod(self, images):
    params = self.config['FastGradientMethod']
    fgm_params = {
      'eps': params['eps'],
      'clip_min': params['clip_min'],
      'clip_max': params['clip_max'],
    }
    tf_imgs = tf.convert_to_tensor(images)
    images_adv = self.sess.run(self.attack.generate(tf_imgs, **fgm_params))
    return images_adv

  def generate_adv(self, images):
    images_adv = self.generate(images)
    if self.config['adv_processing']:
      images_adv *= np.float32(np.int64(images_adv * 255))
      images_adv /= 255
    return images_adv
