PROJECT_PATH="/linkhome/rech/grpgen/ubx31mc/neuralnet"
MODELS_DIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/youtube_models"
CONFIG_PATH="${PROJECT_PATH}/config"
CODE_PATH="${PROJECT_PATH}/code"

DATE=$1
TRAIN_DIR="${MODELS_DIR}/${DATE}"
LOGS_DIR="${MODELS_DIR}/${DATE}_logs"

# export CUDA_VISIBLE_DEVICES='0,1,2,3';
# python3 ${CODE_PATH}/eval.py \
#     --config_file=${LOGS_DIR}/model_flags.yaml \
#     --config_name=eval_train \
#     --train_dir=${TRAIN_DIR} \
#     &>> "${LOGS_DIR}/log_eval_train.logs" &
  

export CUDA_VISIBLE_DEVICES='0,1,2,3'; 
python3 ${CODE_PATH}/eval_youtube.py \
      --config_file=${LOGS_DIR}/model_flags.yaml \
      --config_name=eval_test \
      --train_dir=${TRAIN_DIR} \
      &>> "${LOGS_DIR}/log_eval_test.logs" &

wait

python3 ${CODE_PATH}/parse_events.py \
   --folder=${TRAIN_DIR}

