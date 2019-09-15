
import json
from absl import app, flags

import utils


FLAGS = flags.FLAGS

flags.DEFINE_string("config_file", "params.yaml",
                    "Name of the yaml config file.")
flags.DEFINE_string("config_name", "eval",
                    "Define the execution mode.")
flags.DEFINE_string("train_dir", "",
                    "Name of the training directory")
flags.DEFINE_string("data_dir", "",
                    "Name of the data directory")
flags.DEFINE_bool("start_new_model", False,
                  "Start training a new model or restart an existing one.")
flags.DEFINE_string("backend", "tensorflow",
                    "Wheather run tensorflow model of Pytorch.")
flags.DEFINE_string("params", "",
                    "Parameters to override.")

def main(_):

  if not FLAGS.train_dir or not FLAGS.data_dir:
    raise ValueError("train_dir and data_dir need to be set.")

  params = utils.load_params(
    FLAGS.config_file, FLAGS.config_name, FLAGS.params)
  params.train_dir = FLAGS.train_dir
  params.data_dir = FLAGS.data_dir
  params.start_new_model = FLAGS.start_new_model

  if FLAGS.backend.lower() in ('tensorflow', 'tf'):
    from tensorflow_src.eval import Evaluator
  elif FLAGS.backend.lower() in ('pytorch', 'py', 'torch'):
    from pytorch_src.eval import Evaluator
  else:
    raise ValueError(
      "Backend not recognised. Choose between Tensorflow and Pytorch.")
  evaluate = Evaluator(params)
  evaluate.run()

if __name__ == '__main__':
  app.run(main)

