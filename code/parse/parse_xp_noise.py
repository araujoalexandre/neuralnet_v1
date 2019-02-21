
import sys
import glob
from collections import defaultdict
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
  results = defaultdict(list)
  for folder in folders:
    # get config
    config_file = join('{}_logs'.format(folder), "model_flags.yaml")
    hparams = YParams(config_file, 'train')
    epochs = hparams.num_epochs
    distributions = hparams.wide_resnet['distributions']
    scale_noise = hparams.wide_resnet['scale_noise']
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

    results[distributions].append(
        (basename(folder), epochs, scale_noise, best_acc,
         fgsm_score, carlini_score, pgd_score)
    )
  count = 0
  for key in results.keys():
    res = results[key]
    res = sorted(res, key=lambda x: x[2])
    print('\n{}'.format(key))
    print('folder\tepochs\tscale\tbest_acc\tfgsm_score\tcarlini_score\tpgd_score')
    for x in res:
      print(str(count) + " {}\t{:.0f}\t{:.2f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(*x))
      count += 1


if __name__ == "__main__":
  main()

