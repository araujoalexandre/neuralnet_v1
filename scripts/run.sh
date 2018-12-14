
PROJECT_PATH="/home/aaraujo/neuralnet_structured_matrices"
CONFIG_PATH="${PROJECT_PATH}/config"
CODE_PATH="${PROJECT_PATH}/code"

TRAIN_DIR=$(date '+%Y-%m-%d_%H.%M.%S')
LOGS_DIR="${WORKDIR}/${TRAIN_DIR}_logs"
mkdir ${WORKDIR}/${TRAIN_DIR}
mkdir ${LOGS_DIR}
cp ${CONFIG_PATH}/config.yaml ${LOGS_DIR}/model_flags.yaml

python3 ${CODE_PATH}/train.py \
	--config_file=${CONFIG_PATH}/config.yaml \
	--config_name=train \
	--train_dir=${TRAIN_DIR} \
  &>> "${LOGS_DIR}/log_train.logs" &

# python3 ${CODE_PATH}/eval.py \
# 	--config_file=${CONFIG_PATH}/config.yaml \
# 	--config_name=eval_train \
#   &>> "${LOGS_DIR}/log_eval_train.logs" &
# 
# python3 ${CODE_PATH}/eval.py \
# 	--config_file=${CONFIG_PATH}/config.yaml \
# 	--config_name=eval_test \
#   &>> "${LOGS_DIR}/log_eval_test.logs" &
# 
# wait
# 
# python3 ${CODE_PATH}/parse_events.py \
#    --folder=${WORKDIR}/${TRAIN_DIR}

