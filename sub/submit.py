#!/usr/bin/env python3
import os, sys
import shutil
import json
import argparse
import copy
from itertools import product
from subprocess import Popen, PIPE
from os.path import isdir, exists, join
from datetime import datetime
from jinja2 import Template

_CLUSTER_MAX_TIME_JOB = 20

LIST_ATTACKS = [
  'fgm', 'pgd', 'carlini', 'elasticnet']


class GenerateRunJobConfig:

  def __init__(self, params):

    self.date_format = "%Y-%m-%d_%H.%M.%S_%f"
    self.params = params
    # load template for sbatch script
    with open('sub/template.job') as f:
      self.template = Template(f.read())

    # get environnement variables
    self.params.home = os.environ['HOME']
    self.params.projectdir = os.environ['PROJECTDIR']
    self.params.bash_path = shutil.which("bash")

    # define folder name for training
    if not self.params.debug and not self.params.folder:
      self.params.folder = datetime.now().strftime(
        self.date_format)[:-2]
    elif self.params.debug and not self.params.folder:
      self.params.folder = 'folder_debug'

    if not self.params.debug:
      self.params.with_eval = True
      self.params.job_name = '{}_{}'.format(
        self.params.folder[-4:], self.params.mode)
    else:
      self.params.with_eval = False
      self.params.job_name = 'debug'

    if self.params.cluster == "slurm":
      self.executable = "sbatch"
    elif self.params.cluster == "lsf":
      self.executable = "bsub"
    else:
      self.executable = "bash"

    # if we run the job on a cluster, we may run multiple jobs 
    # to match the time required: if params.time > _CLUSTER_MAX_TIME_JOB, 
    # we run multiple jobs with dependency
    if not self.params.debug:
      if self.params.cluster in ('lsf', 'slurm'):
        njobs = self.params.time // _CLUSTER_MAX_TIME_JOB
        self.times = [60 * _CLUSTER_MAX_TIME_JOB] * njobs
        if self.params.time % _CLUSTER_MAX_TIME_JOB:
          self.times += [(self.params.time % _CLUSTER_MAX_TIME_JOB) * 60]
        self.times = list(map(int, self.times))
      else:
        # we convert the time in minutes
        self.times = [self.params.time * 60]
    else:
      self.times = [60]

    # define file to run if it is not set 
    if not self.params.file:
      if self.params.mode == "train":
        self.params.file = "train"
      elif self.params.mode in ['eval', 'attack']:
        self.params.file = "eval"

    if params.mode == 'train':
      # check if config file exist
      projectdir = os.environ['PROJECTDIR']
      assert exists(
        join(projectdir, 'config', "{}.yaml".format(self.params.config))), \
          "config file '{}' does not exist".format(self.params.config)

    # check if path to model folder exists
    assert isdir(self.params.path), \
        "path '{}' does not exist".format(self.params.path)

    # setup LSF job parameters
    if self.params.cluster != "slurm":
      self.params.ld_library = os.environ['LD_LIBRARY_PATH']
    elif self.params.cluster == "slurm":
      self.params.n_gpus = len(self.params.gpu.split(','))
      if self.params.n_gpus == 1 and not self.params.gpu.split(','):
        self.params.n_gpus = 0

    if self.params.mode == 'train' or \
       (self.params.mode in ('eval', 'attack') and self.params.params):
      self.params.config_folder = 'config_gen'
      self.params.config = self.make_yaml_config()


  def get_name_id(self, outdir):
    id_ = 0
    while exists(join(
      outdir,  'config_{}.yaml'.format(id_))):
      id_ += 1
    return 'config_{}'.format(id_)

  def make_yaml_config(self):
    # if not self.params.name:
    #   raise ValueError("Params is set. Name is are required")
    # load the template and populate the values
    projectdir = os.environ['PROJECTDIR']
    config = open(
      join(projectdir, 'config', '{}.yaml'.format(self.params.config))).read()
    if self.params.params:
      # config = config.format(**json.loads(self.params.params))
      config = config.format(**self.params.params)
    # save new config file in config_gen 
    outdir = join(projectdir, 'config_gen')
    # check if config_gen directory exists in PROJECTDIR
    # create the folder if it does not exist
    if not exists(outdir):
      os.mkdir(outdir)
    # save the config on disk 
    config_name = self.get_name_id(outdir)
    config_path = join(outdir, config_name)
    with open(config_path+'.yaml', "w") as f:
      f.write(config)
    return config_name

  def _execute(self, *args, **kwargs):
    cmd = [self.executable] + list(args)
    return Popen(cmd, stdout=PIPE, stderr=PIPE, **kwargs).communicate()

  def run_job(self, script_id):
    script = self.template.render(**vars(self.params))
    if self.params.cluster == 'None':
      print(script)
      return None
    script_name  = '/tmp/script{}.sh'.format(script_id)
    with open(script_name, 'w') as f:
      f.write(script)
    p = self._execute(script_name)
    result, error = list(map(lambda x: x.decode('utf8'), p))
    if error != '':
      raise RuntimeError("Error in the job submission {}".format(error))
    os.remove(script_name)
    return result

  def _run_training_mode(self):
     # run training
     self.params.start_new_model = True
     for i, time in enumerate(self.times):
       self.params.time = time
       result = self.run_job(i)
       if result is None:
         return self.params.folder
       jobid = result.strip().split(' ')[-1]
       if self.params.cluster == "slurm":
         if "Submitted batch job" in result:
           self.params.start_new_model = False
           self.params.dependency = jobid
       print("Submitted batch job {}".format(jobid))
     # run eval
     if self.params.with_eval:
       self.params.mode = "eval"
       self.params.file = "eval"
       self.params.job_name = '{}_{}'.format(
         self.params.folder[-4:], self.params.mode)
       self.run_job(i+1)
     print("Folder {} created".format(self.params.folder))
     return self.params.folder

  def _run_eval_attack_mode(self):
    self.params.job_name = '{}_{}'.format(
      self.params.folder[-4:], self.params.mode)
    self.params.time = self.times[0]
    result = self.run_job(0)
    if result is not None:
      jobid = result.strip().split(' ')[-1]
      print("Submitted batch job {}".format(jobid))

  def run(self):
    if self.params.mode == "train":
      return self._run_training_mode()
    elif self.params.mode in ('eval', 'attack'):
      return self._run_eval_attack_mode()


def parse_grid_search(params):
  params = params.split(';')
  params = {p.split(':')[0]: p.split(':')[1].split(',') for p in params}
  params = list(dict(zip(params.keys(), values)) \
            for values in product(*params.values()))
  return params

if __name__ == '__main__':

  # default path 
  path = "{}/models".format(os.environ['WORKDIR'])

  parser = argparse.ArgumentParser(
      description='Script to generate bash or slurm script.')
  parser.add_argument("--config", type=str,
                        help="Config file to use for training.")
  parser.add_argument("--folder", type=str,
                        help="Folder of the trained models.")
  parser.add_argument("--mode", type=str, default="train",
                        choices=("train", "eval", "attack"),
                        help="Choose job type train, eval, attack.")
  parser.add_argument("--backend", type=str, default="tensorflow",
                        choices=("tensorflow", "tf", "pytorch", "torch", "py"),
                        help="Choose job type train, eval, attack.")
  parser.add_argument("--with_eval", type=bool, default=True,
                        help="Run the evaluation after training.")
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
  parser.add_argument("--time", type=int, default=20,
                        help="max time for the job")
  parser.add_argument("--dependency", type=int, default=0,
                        help="Defer the start of this job until "
                             "the specified job_id completed.")
  parser.add_argument("--debug", action="store_true",
                        help="Activate debug mode.")

  # parameters for batch experiments
  parser.add_argument("--grid_search", type=str, default='',
            help="Parameters to inject in a template config file.")
  parser.add_argument("--name", type=str, default='',
            help="Name of the batch experiments. Required if grid_search is set.")
  args = vars(parser.parse_args())
  args = parser.parse_args()

  # sanity checks
  if args.mode == 'train' and args.config is None:
    raise ValueError("Train mode needs a config file.")
  if not args.mode in ("train", "eval", "attack"):
    raise ValueError("config not recognized")

  if args.mode == 'eval':
    assert args.folder, \
        "Need to specify the name of the model to evaluate: --folder."
  elif args.mode == 'attack':
    assert args.folder, \
        "Need to specify the name of the model to attack: --folder."
  if args.mode == 'attack':
    assert args.attack, \
        "Need to specify the name of the attack: --attack."
    assert args.attack in LIST_ATTACKS, "Attack not recognized."

  # if debug mode overide time
  if args.debug:
    args.time = 0.1

  if args.grid_search:
    assert args.name, "Required if grid_search is set"
    f_acc = open('script_accuracy_{}.sh'.format(args.name), 'w')
    f_params = open('script_params_{}.sh'.format(args.name), 'w')
    for params in parse_grid_search(args.grid_search):
      args.params = params
      folder = GenerateRunJobConfig(copy.copy(args)).run()
      f_acc.write("echo '{} {}'\n".format(folder, str(params)))
      f_acc.write("cat {}/{}_logs/best_accuracy.txt\n".format(path, folder))
      f_params.write("echo '{}'\n".format(folder))
      f_params.write(
        "python3 utils/inspect_checkpoint.py --folder={}\n".format(folder))
    f_acc.close()
    f_params.close()
  else:
    args.params = None
    GenerateRunJobConfig(args).run()



