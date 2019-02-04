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
    n_layers = hparams.acdc.get('n_layers', None)
    leaky_slope = hparams.acdc.get('leaky_slope', None)
    sign_init = hparams.acdc.get('sign_init', None)
    rand_init = hparams.acdc.get('rand_init', None)
    normal_init = hparams.acdc.get('normal_init', None)
    alpha = hparams.acdc.get('alpha', None)
    ckpts = glob.glob(join(folder, 'model.ckpt*'))
    best_acc_file = join('{}_logs'.format(folder), "best_accuracy.txt")
    best_ckpt, best_acc = 0, None
    if exists(best_acc_file):
      content = open(best_acc_file).readline().strip()
      best_ckpt, best_acc = content.split('\t')

    # print("{}\tlayers:{:3d}\tckpt: {}\tbest ckpt: {: 6d}\tacc: {}".format(
    #   basename(folder), n_layers, int(len(ckpts)/3), int(best_ckpt), best_acc))

    msg = "{}\t"
    for x in ['n_layers', 'leaky_slope', 'sign_init', 'rand_init',
              'normal_init', 'alpha']:
        msg += '{}:'.format(x)
        msg += '{}\t'
    values = [basename(folder), n_layers, leaky_slope,
              sign_init, rand_init, normal_init, alpha]
    print(msg.format(*values))


if __name__ == "__main__":
  main()
