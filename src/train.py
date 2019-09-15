
import json
from absl import app, flags

import utils

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
flags.DEFINE_string("backend", "tensorflow",
                    "Wheather run tensorflow model of Pytorch.")
flags.DEFINE_string("params", None,
                    "Parameters to override.")


def main(_):

  if not FLAGS.train_dir or not FLAGS.data_dir:
    raise ValueError("train_dir and data_dir need to be set.")

  params = utils.load_params(
    FLAGS.config_file, FLAGS.config_name, FLAGS.params)
  params.train_dir = FLAGS.train_dir
  params.data_dir = FLAGS.data_dir
  params.start_new_model = FLAGS.start_new_model
  params.job_name = FLAGS.job_name
  params.ps_hosts = FLAGS.ps_hosts
  params.worker_hosts = FLAGS.worker_hosts
  params.controller_host = FLAGS.controller_host
  params.task_index = FLAGS.task_index
  params.horovod_device = FLAGS.horovod_device

  if FLAGS.backend.lower() in ('tensorflow', 'tf'):
    from tensorflow_src.train import Trainer
  elif FLAGS.backend.lower() in ('pytorch', 'py', 'torch'):
    from pytorch_src.train import Trainer
  else:
    raise ValueError(
      "Backend not recognised. Choose between Tensorflow and Pytorch.")
  trainer = Trainer(params)
  trainer.run()


if __name__ == '__main__':
  app.run(main)


