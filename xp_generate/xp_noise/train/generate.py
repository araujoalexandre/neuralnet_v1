
import os
from os.path import join, exists


def main():

  path = os.path.abspath(os.path.curdir)

  with open('config_template.yaml', 'r') as f:
   config_template = f.read()

  # distributions = [
  #   ("normal", "l2"),
  #   ("exponential", "exp"),
  #   ("weibull", "weibull"),
  #   ("laplace", "l1")
  # ]
  # scales = [
  #   0.01, 0.23, 0.45, 0.68, 0.90,
  #   1.00, 1.34, 1.55, 1.77, 2.00
  # ]
  distributions = [("normal", "l2")]
  scales = [1.00]
  config_names = []
  for name, dist in distributions:
    if not exists(name):
      os.mkdir(name)
    for scale in scales:
      config_name = "config_{}.yaml".format(scale)
      config_names.append(join(path, name, config_name))
      with open(join(name, config_name), 'w') as f:
        f.write(config_template.format(
          distribution=dist, scale=scale))

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
