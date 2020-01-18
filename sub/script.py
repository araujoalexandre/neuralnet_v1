
import os
import shutil
from os.path import join, exists, dirname



class GenerateScript:
  """Create of job script for local or slurm cluster"""

  def __init__(self, 
               cluster=None,
               mode='train',
               train_dir=None,
               job_name=None,
               partition='gpu_p1',
               n_gpus=4,
               n_cpus=30,
               dependency=None,
               file_to_run=None,
               attack_name=None,
               log_filename=None,
               config_file=None,
               backend='tensorflow',
               start_new_model=True,
               override_params=None,
               dev_mode=False,
               distributed_config=None):

    self.cluster = cluster
    self.mode = mode
    self.bash_path = shutil.which("bash")
    self.python_exec = "python3"

    # define path
    self.train_dir = train_dir
    self.models_dir = dirname(train_dir)
    self.logs_dir = '{}_logs'.format(train_dir)
    self.data_dir = os.environ.get('DATADIR')
    self.project_dir = os.environ.get('PROJECTDIR')

    # slrum config
    self.account = os.environ.get('slurm_account', None)
    if self.cluster == "slurm":
      assert self.account is not None, \
        "Slurm account needs to be set in environment variables"
    self.job_name = job_name
    self.partition = partition
    self.n_gpus = n_gpus
    self.n_cpus = n_cpus
    self.dependency = dependency
    self.distributed_config = distributed_config
    self.dev_mode = dev_mode
    
    if self.cluster and self.distributed_config is not None \
      and self.mode == 'train':
      self.distributed = True
      self.nodes = self.distributed_config['nodes']
      self.num_ps = self.distributed_config['num_ps']
      self.num_wk = self.nodes - self.num_ps
      self.srun_worker_id = 0
    else:
      self.distributed = False
      self.nodes = 1

    if backend in ['py', 'pytorch'] and self.distributed:
      # add torch.distributed.launch config before main script
      setup_dist_pytorch = [' -m torch.distributed.launch']
      setup_dist_pytorch.append('--nnodes {}'.format(self.nodes))
      setup_dist_pytorch.append('--node_rank ${node_rank}')
      setup_dist_pytorch.append('--nproc_per_node={}'.format(self.n_gpus))
      setup_dist_pytorch.append('--master_addr ${master_host}')
      setup_dist_pytorch.append('--master_port ${master_port}')
      self.python_exec += ' '.join(setup_dist_pytorch)
    
    # module to load
    self.modules = [
      'cuda/10.1.1',
      'cudnn/10.1-v7.5.1.10',
      'nccl/2.5.6-2-cuda'
    ]

    # python config
    self.file_to_run = file_to_run
    self.log_filename = log_filename
    self.config_file = config_file
    self.attack_name = attack_name
    self.backend = backend
    self.start_new_model = start_new_model
    self.override_params = override_params

    # assertion
    checks = ['job_name', 'file_to_run', 'log_filename']
    for check in checks:
      attr_value = getattr(self, check)
      assert attr_value is not None, \
          '{} needs to be defined'.format(check)

  def switch_to_eval_mode(self):
    self.distributed_config = None
    self.distributed = False
    self.nodes = 1
    self.num_ps = 0
    self.num_wk = self.nodes - self.num_ps
    self.srun_worker_id = 0
    self.log_filename = "eval"
    self.mode = "eval"
    self.file_to_run = "eval.py"
    self.job_name = '{}_{}'.format(self.train_dir[-4:], 'eval')
    self.python_exec = "python3"

  def create_slurm_header(self):
    slurm_header = []
    slurm_header.append('--account={}'.format(self.account))
    slurm_header.append('--job-name={}'.format(self.job_name))
    slurm_header.append('--output={}/slurm_log-%j.out'.format(self.logs_dir))
    slurm_header.append('--error={}/slurm_log-%j.err'.format(self.logs_dir))
    slurm_header.append('--time={}'.format(self.time))
    slurm_header.append('--partition={}'.format(self.partition))
    slurm_header.append('--nodes={}'.format(self.nodes))
    slurm_header.append('--gres=gpu:{}'.format(self.n_gpus))
    # slurm_header.append('--cpus-per-task={}'.format(self.n_cpus))
    slurm_header.append('--wait-all-nodes=1')
    slurm_header.append('--exclusive')
    if self.dependency:
      slurm_header.append('--dependency=afterany:{}'.format(self.dependency))
    if self.dev_mode:
      slurm_header.append('--qos qos_gpu-dev')
    slurm_header = ['{} {}'.format('#SBATCH', arg) for arg in slurm_header]
    slurm_header = '\n'.join(slurm_header)
    return slurm_header

  def create_load_cmd(self):
    modules_load = ['module load {}'.format(module) for module in self.modules]
    load_cmd = 'module purge\n'
    load_cmd += '\n'.join(modules_load) + '\n'
    if self.backend == 'tensorflow':
      load_cmd += 'module load tensorflow-gpu/py3/1.14'
    elif self.backend in ['py', 'pytorch']:
      load_cmd += 'module load pytorch-gpu/py3/1.1'
    return load_cmd

  def unset_proxy(self):
    cmd_unset_proxy = 'unset http_proxy; unset https_proxy; '
    cmd_unset_proxy += 'unset HTTP_PROXY; unset HTTPS_PROXY;'
    return cmd_unset_proxy
    
  def create_distribution_setup(self):
    ps_port = self.distributed_config['ps_port']
    wk_port = self.distributed_config['wk_port']
    setup = [
      'workers_str=$(scontrol show hostname $SLURM_JOB_NODELIST)']
    setup.append('for i in ${workers_str[@]}; do workers+=("$i"); done')

    def _create_hosts(setup, host_type, port, start_id, end_id):
      hosts = ['${{workers[{}]}}:{}'.format(i, port) 
                 for i in range(start_id, end_id)]
      hosts = '"{}"'.format(';'.join(hosts))
      setup.append('{}_hosts={}'.format(host_type, hosts))

    if self.num_ps:
      _create_hosts(setup, 'ps', ps_port, 0, self.num_ps)
    _create_hosts(setup, 'wk', wk_port, self.num_ps, self.nodes)
    return '\n'.join(setup) + '\n'

  def create_srun_cmd(self, job_name=None, task_id=None):
    srun_args = []
    if job_name is None:
      srun_args.append('--output={}/log_{}.logs'.format(
        self.logs_dir, self.log_filename))
    else:
      srun_args.append('--output={}/log_{}_{}_{}.logs'.format(
        self.logs_dir, self.log_filename, job_name, task_id))
    srun_args.append('--unbuffered')
    srun_args.append('--open-mode=append')
    srun_args.append('--nodes=1')
    srun_args.append('--gres=gpu:{}'.format(self.n_gpus))
    # srun_args.append('--cpus-per-task={}'.format(self.n_cpus))
    if self.distributed:
      srun_args.append('--nodelist="${{workers[{}]}}"'.format(
        self.srun_worker_id))
      self.srun_worker_id += 1
    srun_cmd = 'srun {}'.format(' '.join(srun_args)) + ' '
    return srun_cmd

  def create_python_cmd(self, job_name=None, task_id=None):
    args = []
    if self.mode == 'train':
      args.append('--config_file={}'.format(self.config_file))
      args.append('--start_new_model={}'.format(self.start_new_model))
      args.append('--config_name=train')
    if self.mode == 'eval':
      args.append('--config_name=eval')
    elif self.mode == 'attack':
      args.append('--config_name=attack_{}'.format(self.attack_name))
    args.append('--train_dir={}'.format(self.train_dir))
    args.append('--data_dir={}'.format(self.data_dir))
    args.append('--backend={}'.format(self.backend))
    if self.override_params and self.mode in ('eval', 'attack'):
      args.append('--override_params={}'.format(self.override_params))
    if self.distributed:
      args.append('--job_name={}'.format(job_name))
      args.append('--worker_hosts=${wk_hosts}')
      args.append('--ps_hosts=${ps_hosts}')
      args.append('--task_index={}'.format(task_id))

    args = ' '.join(args)
    file_to_run = join(self.project_dir, 'src', self.file_to_run)
    python_cmd = '{} {} {}'.format(self.python_exec, file_to_run, args)
    return python_cmd

  def redirect_logs(self):
    if self.cluster:
      # no redirection
      return ''
    return "&>> {}/log_{}.logs".format(
      self.logs_dir, self.log_filename)

  def generate(self):
    """ Construct a job submission script """
    job_script = '#!{}\n'.format(self.bash_path)
    if self.cluster:
      job_script += '{}\n'.format(self.create_slurm_header())
      job_script += '{}\n'.format(self.create_load_cmd())
      job_script += '{}\n'.format(self.unset_proxy())
    if self.distributed:
      job_script += '{}\n'.format(self.create_distribution_setup())

    if self.cluster and self.distributed:
      if self.backend in ['py', 'pytroch']:
        job_script += "master_host=${workers[0]}\n"
        job_script += "master_port={}\n\n".format(
          self.distributed_config['ps_port'])
        # job_script += 'export NCCL_DEBUG=INFO\n\n'
      if self.num_ps:
        for ps_id in range(self.num_ps):
          job_script += self.create_srun_cmd(
            job_name='ps', task_id=ps_id)
          job_script += '{} &\n'.format(self.create_python_cmd(
            job_name='ps', task_id=ps_id))
      for wk_id in range(self.num_wk):
        job_script += "node_rank={}\n".format(wk_id)
        job_script += self.create_srun_cmd(
          job_name='worker', task_id=wk_id)
        job_script += '{} &\n'.format(self.create_python_cmd(
          job_name='worker', task_id=wk_id))
        job_script += 'pids+=($!)\n'.format(wk_id)
      job_script += 'wait "${pids[@]}"\n'

    else:
      if self.cluster:
        job_script += self.create_srun_cmd()
      job_script += '{}'.format(self.create_python_cmd()) 
      job_script += ' {}'.format(self.redirect_logs())
    return job_script
