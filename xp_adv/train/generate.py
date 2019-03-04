
import os
from os.path import join, exists


def main():

  path = os.path.abspath(os.path.curdir)

  with open('config_template.yaml', 'r') as f:
   config_template = f.read()

  attack_norm = [2]
  params = [
    {'train_with_noise': False, 'learn_noise_defense': False, 'noise_attack': False, 'learn_noise_attack': False},
    {'train_with_noise': True , 'learn_noise_defense': False, 'noise_attack': False, 'learn_noise_attack': False},
    # {'train_with_noise': True , 'learn_noise_defense': False, 'noise_attack': True , 'learn_noise_attack': False},
    # {'train_with_noise': True , 'learn_noise_defense': False, 'noise_attack': True , 'learn_noise_attack': True },
    {'train_with_noise': True , 'learn_noise_defense': True , 'noise_attack': False, 'learn_noise_attack': False},
    # {'train_with_noise': True , 'learn_noise_defense': True , 'noise_attack': True , 'learn_noise_attack': False},
    # {'train_with_noise': True , 'learn_noise_defense': True , 'noise_attack': True , 'learn_noise_attack': True },
  ]

  name_id = 0
  config_names = []
  for norm in attack_norm:
    for param in params:
      param['ord'] = norm
      if param['train_with_noise'] == False:
        param['noise_in_eval'] = False
      else:
        param['noise_in_eval'] = True
      config_name = "config_{}.yaml".format(name_id)
      config_names.append(join(path, 'config', config_name))
      name_id += 1
      with open(join(path, 'config', config_name), 'w') as f:
        f.write(config_template.format(**param))

  with open('train_template.jt', 'r') as f:
    train_template = f.read()

  if not exists(join(path, "sub")):
    os.mkdir(join(path, "sub"))

  run = open('run_all.sh', 'w')
  for i, config_name in enumerate(config_names):
    train_jt = join(path, "sub", "train_{}.jt".format(i))
    with open(train_jt, "w") as f:
      f.write(train_template.format(config=config_name))
      if i != 0: run.write("sleep 2\n")
      run.write("bsub < {}\n".format(train_jt))



if __name__ == '__main__':
  main()
