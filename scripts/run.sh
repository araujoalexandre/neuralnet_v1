
PROJECT_PATH="/home/aaraujo/neuralnet"
CONFIG_PATH="${PROJECT_PATH}/config"
CODE_PATH="${PROJECT_PATH}/code"

if [ "$#" -ne 1 ]
then
  DATE=$(date '+%Y-%m-%d_%H.%M.%S')
  TRAIN_DIR="${WORKDIR}/${DATE}"
  LOGS_DIR="${WORKDIR}/${DATE}_logs"
  # mkdir ${TRAIN_DIR}
  mkdir ${LOGS_DIR}
  cp ${CONFIG_PATH}/config.yaml ${LOGS_DIR}/model_flags.yaml
else
  DATE=$1
  TRAIN_DIR="${WORKDIR}/${DATE}"
  LOGS_DIR="${WORKDIR}/${DATE}_logs"
fi

# reset cuda
export CUDA_VISIBLE_DEVICES='';

python3 ${CODE_PATH}/train.py \
	--config_file=${CONFIG_PATH}/config.yaml \
	--config_name=train \
	--train_dir=${TRAIN_DIR} \
  &>> "${LOGS_DIR}/log_train.logs" &

# python3 ${CODE_PATH}/eval.py \
# 	--config_file=${CONFIG_PATH}/config.yaml \
# 	--config_name=eval_train \
# 	--train_dir=${TRAIN_DIR} \
#   &>> "${LOGS_DIR}/log_eval_train.logs" &

# python3 ${CODE_PATH}/eval.py \
# 	--config_file=${CONFIG_PATH}/config.yaml \
# 	--config_name=eval_test \
# 	--train_dir=${TRAIN_DIR} \
#   &>> "${LOGS_DIR}/log_eval_test.logs" &
# 
# wait
# 
# python3 ${CODE_PATH}/parse_events.py \
#    --folder=${WORKDIR}/${TRAIN_DIR}
# 
