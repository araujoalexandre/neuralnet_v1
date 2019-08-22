import sys
import glob
import pickle
from os.path import join

import numpy as np
from numpy.linalg import norm

def pickle_load(path):
    """function to load pickle object"""
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def get_dataset(path, type):
  files = sorted(glob.glob(join(path, "*{}*.pkl".format(type))))
  array = []
  for file_path in files:
    array.append(pickle_load(file_path))
  return np.vstack(array)

def stats(name, value_norm):
  print(name, value_norm.min(), value_norm.max(), value_norm.mean())

def main():
  folder = sys.argv[1]
  folder_logs = '{}_logs'.format(folder)
  dump_folders = glob.glob(join(folder_logs, "dump*"))
  for dump in dump_folders:
    dump_sample_folder = glob.glob(join(dump, "*"))
    for dump_sample in dump_sample_folder:
      print(dump_sample)
      dump_attack_folder = glob.glob(join(dump_sample, "*"))
      for dump_attack in dump_attack_folder:

        img = get_dataset(dump_attack, type='img')
        adv = get_dataset(dump_attack, type='adv')
        # rescale between 0 and 1
        adv = (adv / 2) + 0.5
        img = (img / 2) + 0.5
        perturbation = adv - img

        perturbation = perturbation.reshape(-1, 299*299*3)
        l1_norm = norm(perturbation, ord=1, axis=1)
        l2_norm = norm(perturbation, ord=2, axis=1)
        linf_norm = norm(perturbation, ord=np.inf, axis=1)

        stats('norm l1', l1_norm)
        stats('norm l2', l2_norm)
        stats('norm linf', linf_norm)

if __name__ == '__main__':
  main()
