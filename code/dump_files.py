
import pickle
from os.path import join, basename, exists, normpath

from config import YParams
from config import hparams as FLAGS


def pickle_dump(file, path):
  """function to dump picke object."""
  with open(path, 'wb') as f:
    pickle.dump(file, f, -1)


class DumpFiles:

  def __init__(self, train_dir):
    self.batch_id = 0
    logs_dir = "{}_logs".format(train_dir)

    attack = FLAGS.attack_method
    sample = FLAGS.attack_sample
    self.img_filename = "dump_{}_img_{}_{}.pkl".format(
      attack, sample, '{}')
    self.adv_filename = "dump_{}_adv_{}_{}.pkl".format(
      attack, sample, '{}')

  def files(self, values):
    img = values['images_batch']
    adv = values['images_adv_batch']
    path_img = join(self.logs_dir, self.img_filename.format(batch_id))
    path_adv = join(self.logs_dir, self.adv_filename.format(batch_id))
    pickle_dump(img, path_img)
    pickle_dump(adv, path_adv)
    self.batch_id += 1
