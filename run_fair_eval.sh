#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=/private/home/laurentmeunier/alex/neuralnet/sample-%j.out
#SBATCH --error=/private/home/laurentmeunier/alex/neuralnet/sample-%j.err
#SBATCH --time=4300
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=20
#SBATCH --get-user-env
#SBTACH --comment="NIPS deadline"

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7';

FOLDERS=$(find ${WORKDIR}/models/* -type d | grep -v "logs" | sort)
for FOLDER in ${FOLDERS}
do
  echo "evaluating ${FOLDER}" >> /private/home/laurentmeunier/alex/neuralnet/eval.logs 
  TRAIN_DIR="${WORDIDR}/${FOLDER}"
  LOGS_DIR="${TRAIN_DIR}_logs"
  CONFIG_FILE="${LOGS_DIR}/model_flags.yaml"
  
  srun -o "$LOGS_DIR/log_eval_test.logs" -u \
    python3 $PROJECTDIR/code/eval.py \
      --config_file=$CONFIG_FILE \
      --config_name=eval_test \
      --train_dir=$TRAIN_DIR \
      --data_dir=$DATADIR \
      --params '{"eval_batch_size": 800}'

  echo "done evaluating ${FOLDER}" >> /private/home/laurentmeunier/alex/neuralnet/eval.logs 
done


