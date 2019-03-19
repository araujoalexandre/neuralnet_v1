
import os
import glob

def main():
  path = os.path.abspath(os.path.curdir)
  with open('eval_attacks_template.jt') as f:
    template = f.read()
  folders = [
    "2019-03-01_02.26.26.8420",
    "2019-03-01_02.26.53.0909",
    "2019-03-01_02.29.07.7236"
  ]
  run = open('run_all.sh', 'w')
  for sample in [20, 50]:
    for i, folder in enumerate(folders):
      file = '{}/sub2/eval_attacks_{}_{}.jt'.format(path, i, sample)
      with open(file, 'w') as f:
        f.write(template.format(folder=folder, sample=sample))
      run.write("bsub < {}\n".format(file))

if __name__ == '__main__':
  main()
