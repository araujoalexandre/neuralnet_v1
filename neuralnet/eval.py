
import json
from os.path import join
from absl import app, flags

import utils


FLAGS = flags.FLAGS

flags.DEFINE_string("config_file", "config.yaml",
                    "Name of the yaml config file.")
flags.DEFINE_string("config_name", "eval",
                    "Define the execution mode.")
flags.DEFINE_string("train_dir", "",
                    "Name of the training directory")
flags.DEFINE_string("data_dir", "",
                    "Name of the data directory")
flags.DEFINE_string("backend", "tensorflow",
                    "Wheather run tensorflow model of Pytorch.")
flags.DEFINE_string("override_params", "",
                    "Parameters to override.")

def main(_):

  if not FLAGS.train_dir or not FLAGS.data_dir:
    raise ValueError("train_dir and data_dir need to be set.")

  config_path = join('{}_logs'.format(FLAGS.train_dir), FLAGS.config_file)
  config_name = FLAGS.config_name
  override_params = FLAGS.override_params

  params = utils.load_params(
    config_path, config_name, override_params=override_params)
  params.train_dir = FLAGS.train_dir
  params.data_dir = FLAGS.data_dir
  params.start_new_model = False

  if FLAGS.backend.lower() in ('tensorflow', 'tf'):
    from neuralnet.tensorflow.eval import Evaluator
  elif FLAGS.backend.lower() in ('pytorch', 'py', 'torch'):
    from neuralnet.pytorch.eval import Evaluator
  else:
    raise ValueError(
      "Backend not recognised. Choose between Tensorflow and Pytorch.")
  evaluate = Evaluator(params)
  evaluate.run()

if __name__ == '__main__':
  app.run(main)

