#!/usr/bin/env python3
import os, sys
import shutil
import json
import argparse
from os.path import isdir, exists, join
from datetime import datetime
from jinja2 import Template

attacks = "fgm pgd carlini elasticnet"
date_format = "%Y-%m-%d_%H.%M.%S_%f"

def get_name_id(outdir, name):
  id_ = 0
  while exists(join(
    outdir,  'config_{}_{}.yaml'.format(name, id_))):
    id_ += 1
  return 'config_{}_{}'.format(name, id_)

def make_config(args):
  if not args.name:
    raise ValueError("Params is set. Name is are required")
  # load the template and populate the values
  projectdir = os.environ['PROJECTDIR']
  template = join(projectdir, 'config', '{}.yaml'.format(args.config))
  with open(template) as f:
   template = f.read()
  config = template.format(**json.loads(args.params))
  # save new config file in config_gen 
  outdir = join(projectdir, 'config_gen')
  # check if config_gen directory exists in PROJECTDIR
  # create the folder if it does not exist
  if not exists(outdir):
    os.mkdir(outdir)
  # save the config on disk 
  config_name = get_name_id(outdir,  args.name)
  config_path = join(outdir, config_name)
  with open(config_path+'.yaml', "w") as f:
    f.write(config)
  return config_name

def main(args):

  # set projectdir
  args.projectdir = os.environ['PROJECTDIR']

  # define folder name for training
  if not args.debug:
    args.date = datetime.now().strftime(date_format)[:-2]
  else:
    args.date = 'folder_debug'

  # define file to run if it is not set 
  if not args.file:
    if args.job_type == "train":
      args.file = "train"
    elif args.job_type in ['eval', 'attack']:
      args.file = "eval"

  # check if config file exist
  projectdir = os.environ['PROJECTDIR']
  assert exists(
    join(projectdir, 'config', "{}.yaml".format(args.config))), \
      "config file '{}' does not exist".format(args.config)

  # check if path to model folder exists
  assert isdir(args.path), \
      "path '{}' does not exist".format(args.path)

  args.home = os.environ['HOME']
  if not args.debug:
    args.job_name = '{}_{}'.format(
      args.date[-4:], args.job_type)
  else:
    args.job_name = 'debug'

  # setup ouessant job parameters 
  if args.cluster != "slurm":
    args.ld_library = os.environ['LD_LIBRARY_PATH']
  elif args.cluster == "slurm":
    args.n_gpus = len(args.gpu.split(','))
    if args.n_gpus == 1 and not args.gpu.split(','):
      args.n_gpus = 0

  # if attack is define
  if args.attack:
    assert args.attack in attacks.split(' '), \
      "Attack not recognized"
    args.attack = "attack_{}".format(args.attack)

  # if params is set, generate config file
  if args.params:
    args.config_folder = 'config_gen'
    args.config = make_config(args)
  else:
    args.config_folder = 'config'

  args.bash_path = shutil.which("bash")

  with open('sub/template.job') as f:
    file = f.read()
  template = Template(file)
  script = template.render(**vars(args))
  print(script)


if __name__ == '__main__':

  # default path 
  path = "{}/models".format(os.environ['WORKDIR'])

  parser = argparse.ArgumentParser(
      description='Run attacks on trained models.')
  parser.add_argument("--config", type=str,
                        help="Config file to use for training.")
  parser.add_argument("--folder", type=str,
                        help="Folder of the trained models.")
  parser.add_argument("--job_type", type=str, default="train",
                        help="Choose job type train, eval, attack.")
  parser.add_argument("--path", type=str, default=path,
                        help="Set path of trained folder.")
  parser.add_argument("--attack", type=str, default='',
                        help="Attack to perform.")
  parser.add_argument("--file", type=str,
                        help="Set file to run")
  parser.add_argument("--cpu", type=int, default=60,
                        help="Set the number of CPU to use.")
  parser.add_argument("--gpu", type=str, default="0,1,2,3",
                        help="Set CUDA_VISIBLE_DEVICES")
  parser.add_argument("--partition", type=str, default="gpu_gct3",
                        help="define the slurm partition to use")
  parser.add_argument("--cluster", type=str, default="slurm",
                        help="slurm to generate slurm srun code, bash otherwise")
  parser.add_argument("--time", type=int, default=1200,
                        help="max time for the job")
  parser.add_argument("--debug", action="store_true",
                        help="Activate debug mode.")

  # paramters for batch experiments
  parser.add_argument("--params", type=str, default='',
            help="Parameters to override in the config file.")
  parser.add_argument("--name", type=str, default='',
            help="Name of the batch experiments. Required if params is set.")
  # args = vars(parser.parse_args())
  args = parser.parse_args()

  # sanity checks
  if args.config is None:
    parser.print_help()
    sys.exit(0)
  if args.job_type == 'eval':
    assert args.folder, \
        "Need to specify the name of the model to evaluate: --folder."
  elif args.job_type == 'attack':
    assert args.folder, \
        "Need to specify the name of the model to attack: --folder."
  if args.job_type == 'attack':
    assert args.attack, \
        "Need to specify the name of the attack: --attack."

  # if debug mode overide time
  if args.debug:
    args.time = 60
  main(args)


