
WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/youtube_models"
PROJECT_PATH="/linkhome/rech/grpgen/ubx31mc/neuralnet"
CODE_PATH="${PROJECT_PATH}/code"

DATE=$1
TRAIN_DIR="${WORKDIR}/${DATE}"
LOGS_DIR="${WORKDIR}/${DATE}_logs"
CONFIG_PATH="${LOGS_DIR}/model_flags.yaml"

# export CUDA_VISIBLE_DEVICES='0,1';
python3 ${CODE_PATH}/train.py \
	--config_file=${CONFIG_PATH} \
	--config_name=train \
	--train_dir=${TRAIN_DIR} \
  &>> "${LOGS_DIR}/log_train.logs" &

# # export CUDA_VISIBLE_DEVICES='2,3';
# python3 ${CODE_PATH}/eval.py \
# 	--config_file=${CONFIG_PATH} \
# 	--config_name=eval_test \
# 	--train_dir=${TRAIN_DIR} \
#   &>> "${LOGS_DIR}/log_eval_test.logs" &
# wait
# 
# python3 ${CODE_PATH}/parse_events.py \
#    --folder=${TRAIN_DIR}
# 
