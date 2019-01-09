
from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams
from tensorflow import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("config_file", "params.yaml",
                    "Name of the yaml config file.")

flags.DEFINE_string("config_name", "train",
                    "Define the execution mode.")

flags.DEFINE_string("train_dir", "auto",
                    "Name of the training directory")

class YParams(HParams):
    def __init__(self, yaml_fn, config_name):
        super().__init__()
        with open(yaml_fn) as fp:
            for k, v in YAML(typ='unsafe').load(fp)[config_name].items():
                self.add_hparam(k, v)

hparams = YParams(FLAGS.config_file, FLAGS.config_name)
# record the train directory
if FLAGS.train_dir != "auto":
  hparams.train_dir = FLAGS.train_dir
