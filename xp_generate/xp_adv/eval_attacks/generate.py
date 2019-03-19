
import os
import glob

def main():
  path = os.path.abspath(os.path.curdir)
  with open('eval_attacks_template.jt') as f:
    template = f.read()
  folders = [
      "2019-02-17_20.21.40.8592","2019-02-17_20.21.40.8597",
  ]
  run = open('run_all.sh', 'w')
  for i, folder in enumerate(folders):
    file = '{}/sub/eval_attacks_{}.jt'.format(path, i)
    with open(file, 'w') as f:
      f.write(template.format(folder=folder))
    run.write("bsub < {}\n".format(file))

if __name__ == '__main__':
  main()
