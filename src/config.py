
import json
from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams
from tensorflow import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("config_file", "params.yaml",
                    "Name of the yaml config file.")
flags.DEFINE_string("config_name", "train",
                    "Define the execution mode.")
flags.DEFINE_string("train_dir", "",
                    "Name of the training directory")
flags.DEFINE_string("data_dir", "",
                    "Name of the data directory")
flags.DEFINE_bool("start_new_model", True,
                  "Start training a new model or restart an existing one.")
flags.DEFINE_enum('job_name', '', ('ps', 'worker', 'controller', ''),
                  'One of "ps", "worker", "controller", "".  Empty for local '
                  'training')
flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of target hosts')
flags.DEFINE_string('worker_hosts', '', 'Comma-separated list of target hosts')
flags.DEFINE_string('controller_host', None, 'optional controller host')
flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
flags.DEFINE_string('horovod_device', '', 'Device to do Horovod all-reduce on: '
                    'empty (default), cpu or gpu. Default with utilize GPU if '
                    'Horovod was compiled with the HOROVOD_GPU_ALLREDUCE '
                    'option, and CPU otherwise.')
flags.DEFINE_string("params", "",
                    "Parameters to override.")


if not FLAGS.train_dir or not FLAGS.data_dir:
  raise ValueError("train_dir and data_dir need to be set.")

def override_params(params, key, new_values):
  if not isinstance(new_values, dict):
    setattr(params, key, new_values)
    return params
  obj = getattr(params, key)
  for k, v in new_values.items():
    obj[k] = v
  setattr(params, key, obj)
  return params

class YParams(HParams):
  def __init__(self, yaml_fn, config_name):
    super().__init__()
    with open(yaml_fn) as fp:
      yaml = YAML(typ='unsafe')
      yaml.allow_duplicate_keys = True
      for k, v in yaml.load(fp)[config_name].items():
        self.add_hparam(k, v)

hparams = YParams(FLAGS.config_file, FLAGS.config_name)
hparams.train_dir = FLAGS.train_dir
hparams.data_dir = FLAGS.data_dir
hparams.start_new_model = FLAGS.start_new_model
hparams.job_name = FLAGS.job_name
hparams.ps_hosts = FLAGS.ps_hosts
hparams.worker_hosts = FLAGS.worker_hosts
hparams.controller_host = FLAGS.controller_host
hparams.task_index = FLAGS.task_index
hparams.horovod_device = FLAGS.horovod_device

# if params is not none, override params
if FLAGS.params:
  params = json.loads(FLAGS.params)
  for key, value in params.items():
    hparams = override_params(hparams, key, value)
