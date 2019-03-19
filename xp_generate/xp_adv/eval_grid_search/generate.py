
import os
from os.path import join

def title(eps):
  eps = str(eps)
  eps = eps.replace('.', '_')
  return eps

def main():
  path = os.path.abspath(os.path.curdir)
  with open('eval_attacks_template.jt') as f:
    template = f.read()
  folders = [
    "2019-03-07_18.00.09.9119",
    "2019-03-07_18.00.34.9387",
    "2019-03-07_18.00.35.5701"
  ]


  run = open('run_all.sh', 'w')
  for i, folder in enumerate(folders):
    for attack in ['fgm', 'pgd']:
      # for norm in [1, 2, 'inf']:
      for norm in ['inf']:
        if attack == 'pgd' and norm == 1:
          continue
        # for eps in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        for eps in [0.3]:
          # for sample in [1, 10, 30, 50]:
          for sample in [50]:
            if folder == '2019-02-27_15.24.06.1345' and sample != 1:
              continue
            config = {'attack':attack, 'eps':title(eps), 'sample':sample, 'norm':norm}
            config_name = 'attack_{attack}_{norm}_{eps}_mc_{sample}'
            config_name = config_name.format(**config)
            filename = join(path, 'sub', '{}_{}.jt'.format(config_name, i))
            with open(filename, 'w') as f:
              job_name = '{}_{}'.format(config_name[7:], i)
              f.write(template.format(
                folder=folder, config_name=config_name, name=job_name))
            run.write("bsub < {}\n".format(filename))

def main_carlini():
  path = os.path.abspath(os.path.curdir)
  with open('eval_attacks_template.jt') as f:
    template = f.read()
  folders = [
    "2019-02-17_20.21.48.5930",
    # "2019-02-27_15.24.06.1345",
    "2019-02-27_15.24.08.0467",
    "2019-03-01_02.26.26.8420"
  ]
  run = open('run_all_carlini.sh', 'w')
  for sample in [10, 30, 50]:
    for i, folder in enumerate(folders):
      config_name = 'attack_carlini_mc_{sample}'.format(sample=sample)
      filename = join(path, 'sub_carlini', '{}_{}.jt'.format(config_name, i))
      with open(filename, 'w') as f:
        job_name = '{}_{}'.format(config_name[7:], i)
        f.write(template.format(
          folder=folder, config_name=config_name, name=job_name))
      run.write("bsub < {}\n".format(filename))

if __name__ == '__main__':
  main()
  # main_carlini()
