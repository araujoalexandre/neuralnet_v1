import sys
import glob
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

def main():
  path = sys.argv[1]
  folders = glob.glob(join(path, "*"))
  folders = list(filter(lambda x: 'logs' not in x, folders))
  folders = sorted(folders)
  for folder in folders:
    # get config
    config_file = join('{}_logs'.format(folder), "model_flags.yaml")
    hparams = YParams(config_file, 'train')
    lr = hparams.piecewise_constant['values'][0]
    rank = hparams.low_rank['rank']
    ckpts = glob.glob(join(folder, 'model.ckpt*'))
    best_acc_file = join('{}_logs'.format(folder), "best_accuracy.txt")
    best_ckpt, best_acc = 0, None
    if exists(best_acc_file):
      content = open(best_acc_file).readline().strip()
      best_ckpt, best_acc = content.split('\t')

    print("{}\trank: {: 3d}\tlr: {}\tckpt: {}\tbest_ckpt: {: 6d}\tacc: {}".format(
      basename(folder), rank, lr, len(ckpts)/3, int(best_ckpt), best_acc))


if __name__ == "__main__":
  main()
