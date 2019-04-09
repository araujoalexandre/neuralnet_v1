#!/pwrlocal/pub/anaconda/py3/bin/python3

import os
import json
import argparse
import socket
from os.path import join
from datetime import datetime

attacks = "fgm pgd carlini"
date_format = "%Y-%m-%d_%H.%M.%S_%f"

setup_ouessant = """
#BSUB -J {folder_id}_train
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

script = """
CONFIG_PATH="$PROJECTDIR/config/{config}.yaml"
TRAIN_DIR="{path}/{date}"
LOGS_DIR="{path}/{date}_logs"
mkdir $LOGS_DIR
cp $CONFIG_PATH $LOGS_DIR"/model_flags.yaml"

export CUDA_VISIBLE_DEVICES='{gpu_train}';
python3 $PROJECTDIR/code/{train}.py \\
  --config_file=$CONFIG_PATH \\
  --config_name=train \\
  --train_dir=$TRAIN_DIR \\
  --data_dir=$DATADIR \\
  &>> $LOGS_DIR"/log_train.logs" &

export CUDA_VISIBLE_DEVICES='{gpu_eval}';
python3 $PROJECTDIR/code/eval.py \\
  --config_file=$CONFIG_PATH \\
  --config_name=eval_test \\
  --train_dir=$TRAIN_DIR \\
  --data_dir=$DATADIR \\
  &>> $LOGS_DIR"/log_eval_test.logs" &
wait
"""

script_attacks = """
export CUDA_VISIBLE_DEVICES='{gpu_attacks}';
for ATTACK in {attacks}
do
  python3 $CODE_PATH/eval.py \\
    --config_file=$CONFIG_FILE \\
    --config_name=$ATTACK \\
    --train_dir=$TRAIN_DIR \\
    --data_dir=$DATADIR \\
    &>> "$LOGS_DIR/log_"$ATTACK".logs" &
  wait
done
"""


def make_config(args):
  if not args['name'] or not args['id']:
    raise ValueError("Params is set. Name and Id are required")
  projetdir = os.environ['PROJECTDIR']
  config_path = join(projetdir, 'config', '{}.yaml'.format(args['config']))
  custom_config_name = '{}_{}.yaml'.format(args['name'], args['id'])
  custom_config_path = join(projetdir, 'config_gen', custom_config_name)
  with open(config_path) as f:
   config = f.read()
  config = config.format(**json.loads(args['params']))
  with open(custom_config_path, "w") as f:
    f.write(config)
  return '{}_{}'.format(args['name'], args['id'])

def main(args):
  global script

  # define folder name for training
  args['date'] = datetime.now().strftime(date_format)[:-2]

  # if train_under_attack set name script
  if args['train'] == "under_attack":
    args['train'] = "train_under_attack"
  else:
    args['train'] = "train"

  # if attacks is define, activate attacks after training
  if args['attacks']:
    script += script_attacks
    args['attacks'] = list(dict.fromkeys(args['attacks'].split(' ')))
    assert set(args['attacks']).issubset(attacks.split(' ')), \
      "attacks not found"
    mapping = lambda x: "'attack_{}'".format(x)
    args['attacks'] = list(map(mapping, args['attacks']))
    args['attacks'] = ' '.join(args['attacks'])

  # if params is set, generate config file
  if args['params']:
    args['config'] = make_config(args)

  # setup ouessant job parameters 
  if "ouessant" in args['hostname']:
    script = setup_ouessant + script
    args['home'] = os.environ['home']
    args['ld_library'] = os.environ['LD_LIBRARY_PATH']
    args['folder_id'] = args['date'][-4:]

  print(script.format(**args))


if __name__ == '__main__':

  # default path 
  path = "{}/models".format(os.environ['WORKDIR'])

  parser = argparse.ArgumentParser(
      description='Run attacks on trained models.')
  parser.add_argument("config", type=str,
                        help="Config file to use for training.")
  parser.add_argument("--path", type=str, default=path,
                        help="Set path of trained folder.")
  parser.add_argument("--attacks", type=str, default='',
                        help="List of attacks to perform.")
  parser.add_argument("--train", type=str,
                        help="Set train scheme.")
  parser.add_argument("--gpu_train", type=str, default="0,1",
                        help="Set CUDA_VISIBLE_DEVICES for training")
  parser.add_argument("--gpu_eval", type=str, default="2,3",
                        help="Set CUDA_VISIBLE_DEVICES for eval.")
  parser.add_argument("--gpu_attacks", type=str, default="0",
                        help="Set CUDA_VISIBLE_DEVICES for attacks.")

  # paramters for batch experiments
  parser.add_argument("--params", type=str, default='',
            help="Parameters to override in the config file.")
  parser.add_argument("--name", type=str, default='',
            help="Name of the batch experiments. Required if params is set.")
  parser.add_argument("--id", type=str, default='',
            help="Id of the experiment. Required if params is set.")
  args = vars(parser.parse_args())

  # get hostname to setup job parameters
  args['hostname'] = socket.gethostname()

  main(args)

