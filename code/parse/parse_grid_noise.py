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
    sensitivity_norm = hparams.random_model['sensitivity_norm']
    layer_sensitivity_bounds = hparams.random_model['layer_sensitivity_bounds'][0]
    attack_norm_bound = hparams.random_model['attack_norm_bound']
    gradient_clip = hparams.gradients['clip_gradient_norm']
    perturb = hparams.gradients['perturbed_gradients']
    ckpts = glob.glob(join(folder, 'model.ckpt-*.index'))
    best_acc_file = join('{}_logs'.format(folder), "best_accuracy.txt")
    best_ckpt, best_acc = 0, None
    if exists(best_acc_file):
      content = open(best_acc_file).readline().strip()
      best_ckpt, best_acc = content.split('\t')

    print("{} {}\{}\tattack_norm: {} ckpt: {: 4d} best_ckpt: {}\tacc: {}\t{}\{}".format(
        basename(folder), sensitivity_norm, layer_sensitivity_bounds,
      attack_norm_bound, len(ckpts), best_ckpt, best_acc, perturb, gradient_clip))


if __name__ == "__main__":
  main()
