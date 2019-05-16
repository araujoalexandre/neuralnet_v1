#!/bin/bash
#SBATCH --job-name=entropy
#SBATCH --output=/private/home/laurentmeunier/alex/neuralnet/sample-%j.out
#SBATCH --error=/private/home/laurentmeunier/alex/neuralnet/sample-%j.err
#SBATCH --time=4300
#SBATCH --partition=PARTITION
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --get-user-env
#SBATCH --comment="NIPS deadline"

TRAIN_DIR="${WORKDIR}/models/FOLDER"
LOGS_DIR=${TRAIN_DIR}"_logs"
CONFIG_FILE=${LOGS_DIR}"/model_flags.yaml"
ENTROPY_LOGS=${LOGS_DIR}"/entropy.logs"

srun -o ${ENTROPY_LOGS} \
  python3 $PROJECTDIR/code/compute_entropy.py \
    --config_file=$CONFIG_FILE \
    --config_name=eval_test \
    --train_dir=$TRAIN_DIR \
    --data_dir=$DATADIR
