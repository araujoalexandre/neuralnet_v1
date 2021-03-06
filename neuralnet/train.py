
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
flags.DEFINE_enum("job_name", "", ("ps", "worker", "controller", ""),
                  "Type of job 'ps', 'worker', 'controller', ''.")
flags.DEFINE_integer("n_gpus", 4, "Number of GPUs to use.")
flags.DEFINE_integer("local_rank", 0, "Define local rank.")
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of target hosts.")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of target hosts.")
flags.DEFINE_string("controller_host", None, "optional controller host.")
flags.DEFINE_string("master_host", "", "ip/hostname of the master.")
flags.DEFINE_string("master_port", "", "port of the master.")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_string("backend", "tensorflow",
                    "Wheather run tensorflow model of Pytorch.")


def main(_):

  if not FLAGS.train_dir or not FLAGS.data_dir:
    raise ValueError("train_dir and data_dir need to be set.")

  params = utils.load_params(FLAGS.config_file, FLAGS.config_name)
  params.train_dir = FLAGS.train_dir
  params.data_dir = FLAGS.data_dir
  params.start_new_model = FLAGS.start_new_model
  params.num_gpus = FLAGS.n_gpus
  params.job_name = FLAGS.job_name
  params.local_rank = FLAGS.local_rank
  params.ps_hosts = FLAGS.ps_hosts
  params.worker_hosts = FLAGS.worker_hosts
  params.controller_host = FLAGS.controller_host
  params.master_host = FLAGS.master_host
  params.master_port = FLAGS.master_port
  params.task_index = FLAGS.task_index

  if FLAGS.backend.lower() == "tensorflow":
    from neuralnet.tensorflow.train import Trainer
  elif FLAGS.backend.lower() == "pytorch":
    from neuralnet.pytorch.train import Trainer
  else:
    raise ValueError(
      "Backend not recognised. Choose between Tensorflow and Pytorch.")
  trainer = Trainer(params)
  trainer.run()


if __name__ == '__main__':
  app.run(main)


