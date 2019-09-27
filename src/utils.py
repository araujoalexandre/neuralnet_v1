
import os
import sys
import re
import shutil
import json
import logging
import glob
import absl.logging
from os.path import join
from os.path import exists

from yaml import load, dump
try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper



def get_global_step_from_ckpt(filename):
  regex = "(?<=ckpt-)[0-9]+"
  return int(re.findall(regex, filename)[-1])


def get_list_checkpoints(train_dir, backend='tensorflow'):
  ext = {'tensorflow': 'index', 'pytorch': 'pth'}[backend]
  files = glob.glob(join(train_dir, 'model.ckpt-*.{}'.format(ext)))
  files = sorted(files, key=get_global_step_from_ckpt)
  if backend == 'tensorflow':
    # we need to remove the extension
    return [filename[:-6] for filename in files]
  return [filename for filename in files]


def get_checkpoint(train_dir, last_global_step, backend='tensorflow'):
  files = get_list_checkpoints(train_dir, backend=backend)
  if not files:
    return None, None
  for filename in files:
    global_step = get_global_step_from_ckpt(filename)
    if last_global_step < global_step:
      return filename, global_step
  return None, None


def get_best_checkpoint(logs_dir, backend='tensorflow'):
  ext = {'tensorflow': 'index', 'pytorch': 'pth'}[backend]
  best_acc_file = join(logs_dir, "best_accuracy.txt")
  if not exists(best_acc_file):
    raise ValueError("Could not find best_accuracy.txt in {}".format(
            logs_dir))
  with open(best_acc_file) as f:
    content = f.readline().split('\t')
    best_ckpt = content[0]
  best_ckpt_path = glob.glob(
    join(logs_dir[:-5], 'model.ckpt-{}.{}'.format(best_ckpt, ext)))
  if backend == 'tensorflow':
    return best_ckpt_path[-1][:-6], int(best_ckpt)
  return best_ckpt_path[-1], int(best_ckpt)




def remove_training_directory(train_dir):
  """Removes the training directory."""
  try:
    if 'debug' in train_dir:
      logging.info(("Train dir already exist, start_new_model "
                    "set to True and debug mode activated. Train folder "
                    "deleted."))
      shutil.rmtree(train_dir)
    else:
      # to be safe we ask the use to delete the folder manually
      raise RuntimeError(("Train dir already exist and start_new_model "
                          "set to True. To restart model from scratch, "
                          "delete the directory."))
  except:
    logging.error("Failed to delete directory {} when starting a new "
      "model. Please delete it manually and try again.".format(train_dir))
    sys.exit()


class MessageBuilder:

  def __init__(self):
    self.msg = []

  def add(self, name, values, align=">", width=0, format=None):
    if name:
      metric_str = "{}: ".format(name)
    else:
      metric_str = ""
    values_str = []
    if type(values) != list:
      values = [values]
    for value in values:
      if format:
        values_str.append("{value:{align}{width}{format}}".format(
          value=value, align=align, width=width, format=format))
      else:
        values_str.append("{value:{align}{width}}".format(
          value=value, align=align, width=width))
    metric_str += '/'.join(values_str)
    self.msg.append(metric_str)

  def get_message(self):
    message = " | ".join(self.msg)
    self.clear()
    return message

  def clear(self):
    self.msg = []



def setup_logging(verbosity):
  formatter = logging.Formatter(
    "[%(asctime)s %(filename)s:%(lineno)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')
  absl.logging.get_absl_handler().setFormatter(formatter)
  log = logging.getLogger('tensorflow')
  level = {'DEBUG': 10, 'ERROR': 40, 'FATAL': 50,
    'INFO': 20, 'WARN': 30
  }[verbosity]
  log.setLevel(level)


class Params:

  def __init__(self):
    self._params = {}

  def add(self, name, value):
    """Adds {name, value} pair to hyperparameters.

    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter. Can be one of the following types:
        int, float, string, int list, float list, or string list.

    Raises:
      ValueError: if one of the arguments is invalid.
    """
    # Keys in kwargs are unique, but 'name' could the name of a pre-existing
    # attribute of this object.  In that case we refuse to use it as a
    # hyperparameter name.
    if getattr(self, name, None) is not None:
      raise ValueError('Hyperparameter name is reserved: %s' % name)
    setattr(self, name, value)
    self._params[name] = value

  def override(self, key, new_values):
    """override a parameter in a Params instance."""
    if not isinstance(new_values, dict):
      setattr(self, key, new_values)
      self._params[key] = new_values
    obj = getattr(self, key)
    for k, v in new_values.items():
      obj[k] = v
    setattr(self, key, obj)
    self._params[key] = obj

  def values(self):
    return self._params

  def to_json(self, indent=None, separators=None, sort_keys=False):
    """Serializes the hyperparameters into JSON.
    Args:
      indent: If a non-negative integer, JSON array elements and object members
        will be pretty-printed with that indent level. An indent level of 0, or
        negative, will only insert newlines. `None` (the default) selects the
        most compact representation.
      separators: Optional `(item_separator, key_separator)` tuple. Default is
        `(', ', ': ')`.
      sort_keys: If `True`, the output dictionaries will be sorted by key.
    Returns:
      A JSON string.
    """
    return json.dumps(
        self.values(),
        indent=indent,
        separators=separators,
        sort_keys=sort_keys)


def load_params(config_file, config_name, override_params):
  params = Params()
  # yaml = YAML()
  with open(config_file) as f:
    data = load(f, Loader=Loader)
  for k, v in data[config_name].items():
    params.add(k, v)
  if override_params:
    params_to_override = json.loads(override_params)
    for key, value in params_to_override.items():
      params.override(key, value)
  return params
