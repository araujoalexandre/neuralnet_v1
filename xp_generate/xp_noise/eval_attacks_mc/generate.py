
import os
import glob

def main():
  path = os.path.abspath(os.path.curdir)
  with open('eval_attacks_template.jt') as f:
    template = f.read()
  folders = [
    "2019-02-17_20.21.48.5930",
    "2019-02-18_17.21.40.8891",
    "2019-02-18_11.45.24.7648",
  ]
  run = open('run_all.sh', 'w')
  for sample in [20, 50, 100]:
    for i, folder in enumerate(folders):
      file = '{}/sub/eval_attacks_{}_{}.jt'.format(path, i, sample)
      with open(file, 'w') as f:
        f.write(template.format(folder=folder, sample=sample))
      run.write("bsub < {}\n".format(file))

if __name__ == '__main__':
  main()
