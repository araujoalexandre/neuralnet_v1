"""A simple script for inspect checkpoint files."""

import argparse
import sys
import re
import os
import glob
from os.path import join, splitext

import numpy as np


def inspect_tensorflow_params(checkpoint_name, full=False):
  """Prints number of parameters in a TensorFlow checkpoint file.

  Args:
    checkpoint_name: Name of the checkpoint file.
  """
  from tensorflow.python import pywrap_tensorflow
  reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_name)
  var = reader.debug_string().decode("utf-8")
  total_params = 0
  filtered_var = ''
  for line in var.split('\n'):
    if 'adam' not in line.lower() and 'momentum' not in line.lower():
      filtered_var += '{}\n'.format(line)
  for params in re.findall('\[[0-9,]+\]', filtered_var):
    total_params += np.prod([int(x) for x in re.findall('[0-9]+', params)])
  if full:
    print(filtered_var)
  print('total parameters = {}'.format(total_params))


def inspect_pytorch_checkpoint(checkpoint_name, full=False):
  """Prints number of parameters in a Pytorch checkpoint file.

  Args:
    checkpoint_name: Name of the checkpoint file.
  """
  import torch
  total_params = 0
  tensor_dict = torch.load(checkpoint_name, map_location='cpu')
  for key, tensor in tensor_dict['model_state_dict'].items():
    if full:
      print(key, tensor.numpy().shape, tensor.numpy().size)
    total_params += tensor.numpy().size
  print('\ntotal parameters = {}'.format(total_params))



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--folder", type=str,
                        help="Folder of train dir model.")
  parser.add_argument("--folder_workdir", type=str, default="models",
                        help="Folder of train dir model.")
  parser.add_argument("--full", action="store_true",
                        help="Print all tensor.")
  parser.add_argument("--ckpt_name", type=str, default="",
                        help="Name of the checkpoint")
  args = parser.parse_args()

  if not args.folder:
    print("Usage: inspect_checkpoint --folder=train_dir_folder")
    sys.exit(1)

  workdir = os.environ.get('WORKDIR', None)
  if workdir is None:
    print("Need to set workdir in envionnement or a --ckpt_name")
    sys.exit(1)

  args.folder = join(workdir, args.folder_workdir, args.folder, 'model.ckpt-0')
  ckpt_full_path = glob.glob(args.folder+'*')[0]
  _, file_extension = splitext(ckpt_full_path)
  if file_extension == ".pth":
    inspect_pytorch_checkpoint(args.folder+'.pth', full=args.full)
  else:
    inspect_tensorflow_params(args.folder, full=args.full)




