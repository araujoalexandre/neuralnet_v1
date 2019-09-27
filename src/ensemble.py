
import os
import json
from os.path import join
from absl import app, flags

import utils

FLAGS = flags.FLAGS
flags.DEFINE_string("folders", "",
                    "Models to make ensemble with.")
flags.DEFINE_string("backend", "tensorflow",
                    "Wheather run tensorflow model of Pytorch.")

def main(_):

  # default path 
  path = "{}/models".format(os.environ['WORKDIR'])

  # expend folders path
  folders = FLAGS.folders.split(';')
  folders = ['{}/{}'.format(path, folder) for folder in folders]

  # load config from first folder
  config_file = join(folders[0]+'_logs', 'model_flags.yaml')
  params = utils.load_params(config_file, 'eval', None)

  params.folders = folders

  # get data dir
  params.data_dir = os.environ['DATADIR']

  if FLAGS.backend.lower() in ('tensorflow', 'tf'):
    from tensorflow_src.ensemble import EvaluatorEnsemble
  elif FLAGS.backend.lower() in ('pytorch', 'py', 'torch'):
    from pytorch_src.ensemble import EvaluatorEnsemble
  else:
    raise ValueError(
      "Backend not recognised. Choose between Tensorflow and Pytorch.")
  evaluate = EvaluatorEnsemble(params)
  evaluate.run()

if __name__ == '__main__':
  app.run(main)





