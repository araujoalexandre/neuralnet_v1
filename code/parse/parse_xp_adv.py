
import sys
import glob
from collections import defaultdict, OrderedDict
from os.path import join, basename, exists
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from tensorflow.contrib.training import HParams
from tensorflow import flags

class YParams(HParams):
    def __init__(self, yaml_fn, config_name):
        super().__init__()
        with open(yaml_fn) as fp:
            for k, v in YAML(typ='unsafe').load(fp)[config_name].items():
                self.add_hparam(k, v)

def get_score_under_attack(path, filename):
  files = glob.glob(
    join(path, "score_{}_{}".format(filename, "*")))
  if len(files) > 0:
    filename = files[0]
    with open(filename) as f:
      content = f.read().strip().split('\t')[1]
    score = content.strip()
    return float(score)
  return -1.

def main():
  path = sys.argv[1]
  folders = glob.glob(join(path, "*"))
  folders = list(filter(lambda x: 'logs' not in x, folders))
  folders = sorted(folders)
  results = []
  for i, folder in enumerate(folders):
    # get config
    config_file = join('{}_logs'.format(folder), "model_flags.yaml")
    hparams = YParams(config_file, 'train')
    if getattr(hparams, "train_attack", None) is None:
      continue
    ckpts = glob.glob(join(folder, 'model.ckpt*'))

    log_folder = '{}_logs'.format(folder)
    # get best accuracy 
    best_acc_file = join(log_folder, "best_accuracy.txt")
    best_ckpt, best_acc = 0, -1.
    if exists(best_acc_file):
      content = open(best_acc_file).readline().strip()
      best_ckpt, best_acc = content.split('\t')
      best_acc = float(best_acc)

    fgsm_score = get_score_under_attack(
      '{}_logs'.format(folder), 'FastGradientMethod')
    carlini_score = get_score_under_attack(
      '{}_logs'.format(folder), 'Carlini')
    pgd_score = get_score_under_attack(
      '{}_logs'.format(folder), 'ProjectedGradientDescent')

    r = OrderedDict(
      folder_id=i,
      folder=basename(folder),
      best_acc=best_acc,
      norm=hparams.train_attack['ProjectedGradientDescent']['ord'],
      pgd_eps=hparams.train_attack['ProjectedGradientDescent']['eps'],
      pgd_eps_iter=hparams.train_attack['ProjectedGradientDescent']['eps_iter'],
      pgd_iter=hparams.train_attack['ProjectedGradientDescent']['nb_iter'],
      train_with_noise=hparams.wide_resnet['train_with_noise'],
      learn_noise_defense=hparams.wide_resnet['learn_noise_defense'],
      noise_attack=hparams.train_attack['noise_attack'],
      learn_noise_attack=hparams.train_attack['learn_noise_attack'],
      fgsm_score=fgsm_score,
      carlini_score=carlini_score,
      pgd_score=pgd_score
    )
    results.append(r)

  message = ("{folder_id}\t{folder}\t{best_acc:.3f}\t"
             "{norm}\t{pgd_eps}\t{pgd_eps_iter}\t{pgd_iter}\t"
             "{train_with_noise}\t{learn_noise_defense}\t{noise_attack}\t{learn_noise_attack}\t"
             "{fgsm_score:.3f}\t{carlini_score:.3f}\t{pgd_score:.3f}")

  print('\t'.join(results[0].keys()))
  for res in results:
    print(message.format(**res))




if __name__ == "__main__":
  main()

