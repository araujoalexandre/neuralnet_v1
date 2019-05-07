#!/usr/bin/env python3

import os
import json
import socket
import argparse
from os.path import join, exists

setup_ouessant = """
#BSUB -J {folder_id}_eval
#BSUB -gpu "num=4:mode=exclusive_process:mps=no:j_exclusive=yes"
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -W 1200
#BSUB -o {home}/%J.train.out
#BSUB -e {home}/%J.train.err

ml anaconda/py3 cudnn nccl
source activate tensorflow1.12-py3
export LD_LIBRARY_PATH={ld_library}
"""


setup_fair = """#!/usr/bin/bash
#SBATCH --job-name={folder_id}_eval
#SBATCH --output={home}/neuralnet/sample-%j.out
#SBATCH --error={home}/neuralnet/sample-%j.err
#SBATCH --time=4300
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --get-user-env

TRAIN_DIR="{path}/{folder}"
LOGS_DIR="$TRAIN_DIR_logs"
CONFIG_FILE="$LOGS_DIR/model_flags.yaml"

export CUDA_VISIBLE_DEVICES='{gpu}';
srun -o "$LOGS_DIR/log_eval_test.logs" -u \\
  --nodes=1 \\
  --gres=gpu:{n_gpus} \\
  --cpus-per-task=10 \\
  python3 $PROJECTDIR/code/eval.py \\
    --config_file=$CONFIG_FILE \\
    --config_name=eval_test \\
    --train_dir=$TRAIN_DIR \\
    --data_dir=$DATADIR {params}
"""

script = """
TRAIN_DIR="{path}/{folder}"
LOGS_DIR="$TRAIN_DIR_logs"
CONFIG_FILE="$LOGS_DIR/model_flags.yaml"

export CUDA_VISIBLE_DEVICES='{gpu}';
python3 $PROJECTDIR/code/eval.py -u \\
  --config_file=$CONFIG_FILE \\
  --config_name=eval_test \\
  --train_dir=$TRAIN_DIR \\
  --data_dir=$DATADIR {params} \\
  &>> "$LOGS_DIR/log_eval_test.logs" &
"""

def main(args):
  global script

  train_dir = join(args['path'], args['folder'])
  assert exists(train_dir), "{} does not exist".format(train_dir)

  # if params is set, overide the parameters in the config file
  if args['params']:
    try:
      _ = json.loads(args['params'])
      args.params = "--params '{}'".format(args['params'])
    except:
      raise ValueError("Could not parse overide parameters")

  # setup ouessant job parameters 
  if "ouessant" in args['hostname']:
    script = setup_ouessant + script
    args['home'] = os.environ['home']
    args['ld_library'] = os.environ['LD_LIBRARY_PATH']
    args['folder_id'] = args['folder'][-4:]
  elif "fair" in args['hostname']:
    script = setup_fair
    args['home'] = os.environ['home']
    args['folder_id'] = args['folder'][-4:]
    args['n_gpus'] = len(args['gpu'].split(','))
  print(script.format(**args))


if __name__ == '__main__':

  # default path 
  path = "{}/models".format(os.environ['WORKDIR'])

  parser = argparse.ArgumentParser(
      description='Run attacks on trained models.')
  parser.add_argument("folder", type=str,
                        help="Folder of the trained models.")
  parser.add_argument("--path", type=str, default=path,
                        help="path of the trained folder.")
  parser.add_argument("--gpu", type=str, default="0,1,2,3",
                        help="Set CUDA_VISIBLE_DEVICES for eval.")
  parser.add_argument("--partition", type=str, default="learnfair",
                       help="define the patition to use. FAIR only.")
  # paramters for batch experiments
  parser.add_argument("--params", type=str, default='',
            help="Parameters to override in the config file.")
  parser.add_argument("--name", type=str, default='',
            help="Name of the batch experiments. Required if params is set.")
  args = vars(parser.parse_args())

  # get hostname to setup job parameters
  args['hostname'] = socket.gethostname()
  main(args)


