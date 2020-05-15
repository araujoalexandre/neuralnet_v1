#!/usr/bin/env python3
import os, sys
import shutil
import json
import argparse
import tempfile
import time
from itertools import product
from subprocess import Popen, PIPE
from os.path import isdir, exists, join
from os.path import basename, dirname
from datetime import datetime

from script import GenerateScript

_CLUSTER_MAX_TIME_JOB = 20

LIST_ATTACKS = [
  'fgm', 'fgsm', 'pgd', 'pgd_l2', 'pgd_linf', 'carlini', 'elasticnet']

DATE_FORMAT = "%Y-%m-%d_%H.%M.%S_%f"

def copytree(src, dst, symlinks=False, ignore=None):
  for item in os.listdir(src):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if os.path.isdir(s):
      shutil.copytree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)

class GenerateRunJobConfig:

  def __init__(self,
               config_file=None,
               models_dir=None,
               train_dir=None,
               mode=None,
               backend='tensorflow',
               with_eval=None,
               start_new_model=None,
               attack_name=None,
               file_to_run=None,
               n_cpus=None,
               n_gpus=None,
               partition=None,
               cluster=None,
               time=None,
               dependency=None,
               template_config_params=None,
               override_params=None,
               debug=False,
               dev=False,
               verbose=False,
               distributed_config=None):

    self.config_file = '{}.yaml'.format(config_file)
    self.models_dir = models_dir
    self.train_dir = train_dir
    self.mode = mode
    self.backend = backend
    self.with_eval = with_eval
    self.start_new_model = start_new_model
    self.attack_name = attack_name
    self.file_to_run = file_to_run
    self.n_cpus = n_cpus
    self.n_gpus = n_gpus
    self.slurm_partition = partition
    self.cluster = cluster
    self.time = time
    self.dependency = dependency
    self.template_config_params = template_config_params
    self.override_params = override_params
    self.debug = debug
    self.dev = dev
    self.verbose = verbose
    self.dev_or_debug_mode = self.debug or self.dev
    self.distributed_config = distributed_config

    if self.cluster == "slurm":
      self.executable = 'sbatch'
    else:
      self.executable = 'bash'
    
    self.projectdir = os.environ.get('PROJECTDIR', None)
    assert self.projectdir is not None, \
      "You need to define a project directory in environement variable"

    # define the name of the log file
    if self.mode == 'train':
      self.log_filename = 'train'
    elif self.mode in 'eval':
      self.log_filename = 'eval'
    elif self.mode == 'attack':
      self.log_filename = 'attack_{}'.format(self.attack_name)

    # define folder name for training
    if not self.dev_or_debug_mode and not self.train_dir:
      self.train_dir = join(
        models_dir, datetime.now().strftime(DATE_FORMAT)[:-2])
    elif self.dev_or_debug_mode and not self.train_dir:
      self.train_dir = join(models_dir, 'folder_debug')
    else:
      self.train_dir = join(models_dir, self.train_dir)
    
    self.logs_dir  = '{}_logs'.format(self.train_dir)
    # we don't create the train directory because it will be created by the
    # train session, we remove folder_debug_logs before creating it again
    if self.dev_or_debug_mode and exists(self.logs_dir):
      shutil.rmtree(self.logs_dir)
      os.mkdir(self.logs_dir)
    elif not exists(self.logs_dir):
      os.mkdir(self.logs_dir)

    if not self.dev_or_debug_mode:
      self.job_name = '{}_{}'.format(self.train_dir[-4:], self.mode)
    else:
      self.with_eval = False
      self.job_name = 'debug'

    # if we run the job on a cluster, we may run multiple jobs 
    # to match the time required: if self.time > _CLUSTER_MAX_TIME_JOB, 
    # we run multiple jobs with dependency
    if not self.dev_or_debug_mode:
      if self.cluster == 'slurm':
        njobs = self.time // _CLUSTER_MAX_TIME_JOB
        self.times = [60 * _CLUSTER_MAX_TIME_JOB] * njobs
        if self.time % _CLUSTER_MAX_TIME_JOB:
          self.times += [(self.time % _CLUSTER_MAX_TIME_JOB) * 60]
        self.times = list(map(int, self.times))
      else:
        # not cluster job, unlimitied time
        self.times = None
    else:
      # in dev or debug mode, we only ask for 60 minutes
      self.times = [60]

    # define file to run if it is not set 
    if not self.file_to_run:
      if self.mode == "train":
        self.file_to_run = "train.py"
      elif self.mode in ['eval', 'attack']:
        self.file_to_run = "eval.py"

    if self.mode == 'train':
      self.config_path = self.make_yaml_config()
      # copy the src code into config folder
      src_folder = join(self.projectdir, 'neuralnet')
      if not exists(join(self.logs_dir, 'neuralnet')):
        copytree(src_folder, join(self.logs_dir, 'neuralnet'))

    elif self.mode in ('eval', 'attack'):
      self.config_path = join(self.logs_dir, 'config.yaml')
      assert exists(self.config_path), \
          "config.yaml not found in '{}'.".format(basename(self.train_dir))

    # init script template object
    self.script_template = GenerateScript(
                        cluster=self.cluster,
                        mode=self.mode,
                        train_dir=self.train_dir,
                        job_name=self.job_name,
                        partition=self.slurm_partition,
                        n_gpus=self.n_gpus,
                        n_cpus=self.n_cpus,
                        dependency=self.dependency,
                        file_to_run=self.file_to_run,
                        attack_name=self.attack_name,
                        override_params=self.override_params,
                        log_filename=self.log_filename,
                        config_file=self.config_path,
                        backend=self.backend,
                        start_new_model=self.start_new_model,
                        dev_mode=self.dev,
                        distributed_config=self.distributed_config)


  def make_yaml_config(self):
    # load the template and populate the values
    if self.backend in ["tensorflow", "tf"]:
      config_path = join(
        self.projectdir, 'config', 'tensorflow', self.config_file)
    else:
      config_path = join(
        self.projectdir, 'config', 'pytorch', self.config_file)
    assert exists(config_path), \
          "config file '{}' does not exist".format(self.config_file)
    with open(config_path) as f:
      config = f.read()
    if getattr(self, 'template_config_params', None):
      config = config.format(**self.template_config_params)
    # save the config on disk 
    config_path = join(self.logs_dir, 'config.yaml')
    with open(config_path, "w") as f:
      f.write(config)
    return config_path

  def _execute(self, *args, **kwargs):
    cmd = [self.executable] + list(args)
    return Popen(cmd, stdout=PIPE, stderr=PIPE, **kwargs).communicate()

  def run_job(self, script):
    with tempfile.NamedTemporaryFile(mode='w') as fp:
      fp.write(script)
      fp.flush()
      if self.verbose:
        print(script)
      p = self._execute(fp.name)
      # p = (bytes('Submitted batch job 1234'.encode('utf8')), bytes(''.encode('utf8')))
    if self.cluster is not None:
      result, error = list(map(lambda x: x.decode('utf8'), p))
      if error:
        raise RuntimeError("Error in the job submission {}".format(error))
      return result 
    return None

  def run_training_mode(self):
    # run training jobs
    # we may run several dependent job if time > MAX_CLUSTER_TIME
    jobids = []
    for time in self.times:
      self.script_template.time = time 
      script = self.script_template.generate()
      result = self.run_job(script)
      if result is None:
        return self.train_dir
      jobid = result.strip().split(' ')[-1]
      jobids.append(jobid)
      if self.cluster == "slurm":
        if "Submitted batch job" in result:
          self.script_template.start_new_model = False
          self.script_template.dependency = jobid
    # run eval
    if self.with_eval:
      self.script_template.switch_to_eval_mode()
      script = self.script_template.generate()
      result = self.run_job(script)
      if result is not None:
        jobid = result.strip().split(' ')[-1]
        jobids.append(jobid)

    print("Submitted batch job {}".format(' '.join(jobids)))
    print("Folder '{}' created".format(basename(self.train_dir)))
    return self.train_dir, jobids

  def run_eval_attack_mode(self):
    self.script_template.time = self.times[0]
    script = self.script_template.generate()
    result = self.run_job(script)
    if result is not None:
      jobid = result.strip().split(' ')[-1]
      print("Submitted batch job {}".format(jobid))

  def run(self):
    if self.mode == "train":
      return self.run_training_mode()
    elif self.mode in ('eval', 'attack'):
      return self.run_eval_attack_mode()


class GridSearchUtils:
  """Create utils scripts for grid search experiment."""

  def __init__(self, xp_name, debug=False):

    self.debug = debug
    if not debug:

      # bash script to get parameters of all models
      filename1 = 'script_accuracy_{}.sh'.format(xp_name)
      assert not exists(filename1)
      self.file1 = open(filename1, 'w')

      # bash script to scancel all jobs
      filename3 = 'script_scancel_jobs_{}.sh'.format(xp_name)
      assert not exists(filename3)
      self.file3 = open(filename3, 'w')

      # bash script to print all jobs
      filename4 = 'script_squeue_jobs_{}.sh'.format(xp_name)
      assert not exists(filename4)
      self.file4 = open(filename4, 'w')

      self.all_jobids = []


  def write(self, folder, params, jobids):
    if not self.debug:
      self.all_jobids.extend(jobids)
      self.file1.write("echo '{} {}'\n".format(folder, str(params)))
      self.file1.write("cat {}_logs/best_accuracy.txt\n".format(folder))
      self.file3.write("scancel {}\n".format(' '.join(jobids)))

  def close(self):
    if not self.debug:
      self.file1.close()
      self.file3.close()
      self.file4.write('squeue -j {}'.format(','.join(self.all_jobids)))
      self.file4.close()



def parse_grid_search(params):
  params = params.split(';')
  params = {p.split(':')[0]: p.split(':')[1].split(',') for p in params}
  params = list(dict(zip(params.keys(), values)) \
            for values in product(*params.values()))
  return params

if __name__ == '__main__':

  project_dir = os.environ.get('PROJECTDIR', None)
  assert project_dir is not None, \
    "A project directory needs to be set in environnement variables"
  work_dir = os.environ.get('WORKDIR', None)
  assert work_dir is not None, \
    "A working directory needs to be set in environnement variables"
  data_dir = os.environ.get('DATADIR', None)
  assert data_dir is not None, \
    "A project directory needs to be set in environnement variables"

  parser = argparse.ArgumentParser(
      description='Script to generate bash or slurm script.')
  parser.add_argument("--config", type=str,
                        help="Name of the config file to use for training.")
  parser.add_argument("--train_dir", type=str,
                        help="Name or path of train directory.")
  parser.add_argument("--mode", type=str, default="train",
                        choices=("train", "eval", "attack"),
                        help="Choose job type train, eval, attack.")
  parser.add_argument("--backend", type=str, default="pytorch",
                        choices=("tensorflow", "tf", "pytorch", "torch", "py"),
                        help="Choose job type train, eval, attack.")
  parser.add_argument("--no_eval", action="store_true", default=False,
                        help="Run the evaluation after training.")
  parser.add_argument("--restart_model", action="store_true", default=False,
                        help="Start the training of a new model or restart one.")
  parser.add_argument("--models_dir", type=str,
                        help="Set path of trained folders.")
  parser.add_argument("--attack", type=str, default='',
                        help="Attack to perform.")
  parser.add_argument("--file_to_run", type=str,
                        help="Set file to run")
  parser.add_argument("--n_cpus", type=int, default=40,
                        help="Number of cpu to use.")
  parser.add_argument("--n_gpus", type=int, default=4,
                        help="Number of gpu to use.")
  parser.add_argument("--partition", type=str, default="gpu_p1",
                        help="Define the slurm partition to use")
  parser.add_argument("--cluster", type=str, default="slurm",
                        help="slurm to generate slurm srun code, bash otherwise")
  parser.add_argument("--time", type=int, default=20,
                        help="max time for the job")
  parser.add_argument("--dependency", type=int, default=0,
                        help="Defer the start of this job until "
                             "the specified job_id completed.")
  parser.add_argument("--debug", action="store_true",
                        help="Activate debug mode.")
  parser.add_argument("--dev", action="store_true",
                        help="Activate dev mode.")
  parser.add_argument("--verbose", action="store_true",
                        help="Activate print information.")

  # parameters for batch experiments
  parser.add_argument("--grid_search", type=str, default='',
            help="Parameters to inject in a template config file.")
  parser.add_argument("--name", type=str, default='',
            help="Name of the batch experiments. Required if grid_search is set.")

  # argument for overriding parameters for evaluation or attacks
  parser.add_argument("--override_params", type=str, default='',
                      help="overriding parameters for evaluation or attacks")
  
  # parameters for distributed settings
  parser.add_argument("--nodes", type=int, default=1,
                      help="Number of nodes to use for cluster job.")
  parser.add_argument("--num_ps", type=int, default=0,
                      help="Number of parameter server to use for distributed jobs.")
  parser.add_argument("--ps_port", type=int, default=9001,
                      help="Port to use for parameter server.")
  parser.add_argument("--wk_port", type=int, default=9000,
                      help="Port to use for workers." )
 
  # parse all arguments 
  args = parser.parse_args()


  if args.nodes > 1 and args.mode == 'train':
    assert args.num_ps <= args.nodes // 2, \
        "Number of parameter server seems to high."
    distributed_config = {
      "nodes": args.nodes,
      "num_ps": args.num_ps,
      "ps_port": args.ps_port,
      "wk_port": args.wk_port
    }
  else:
    distributed_config = None


  # define or create a models directory in workdir
  if args.models_dir is None:
    args.models_dir = join(work_dir, "models")
    if not exists(args.models_dir):
      os.mkdir(args.models_dir)

  # sanity checks
  if args.mode == 'train' and args.config is None:
    raise ValueError("Train mode needs the name of a config file.")
  if not args.mode in ("train", "eval", "attack"):
    raise ValueError(
      "mode not recognized, should be ('train', 'eval', 'attack')")

  if args.mode == 'eval':
    assert args.train_dir, \
        "Need to specify the name of the model to evaluate: --folder."
  elif args.mode == 'attack':
    assert args.train_dir, \
        "Need to specify the name of the model to attack: --folder."
    assert args.attack, \
        "Need to specify the name of the attack: --attack."
    assert args.attack in LIST_ATTACKS, "Attack not recognized."

  if args.grid_search:
    assert args.name, "Required if grid_search is set"
  
  job_params = dict(
    config_file=args.config,
    train_dir=args.train_dir,
    mode=args.mode,
    backend=args.backend,
    with_eval=not args.no_eval,
    start_new_model=not args.restart_model,
    models_dir=args.models_dir,
    attack_name=args.attack,
    file_to_run=args.file_to_run,
    n_cpus=args.n_cpus,
    n_gpus=args.n_gpus,
    partition=args.partition,
    cluster=args.cluster,
    time=args.time,
    dependency=args.dependency,
    override_params=args.override_params,
    debug=args.debug,
    dev=args.dev,
    verbose=args.verbose,
    distributed_config=distributed_config)

  if args.grid_search:
    grid_search_utils = GridSearchUtils(args.name, debug=args.debug)
    for params in parse_grid_search(args.grid_search):
      print(params)
      job = GenerateRunJobConfig(
        template_config_params=params, **job_params)
      folder, jobids = job.run()
      time.sleep(0.1)
      grid_search_utils.write(folder, params, jobids)
    grid_search_utils.close()
  else:
    job = GenerateRunJobConfig(**job_params)
    job.run()



