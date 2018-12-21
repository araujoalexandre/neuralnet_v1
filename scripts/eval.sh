PROJECT_PATH="/home/aaraujo/neuralnet_structured_matrices"
MODELS_DIR="/srv/osirim/aaraujo/neuralnet_structured_matrices/grid_search_sgd"
CONFIG_PATH="${PROJECT_PATH}/config"
CODE_PATH="${PROJECT_PATH}/code"

DATE=$1
TRAIN_DIR=${MODELS_DIR}/${TRAIN_DIR}
LOGS_DIR="${MODELS_DIR}/${TRAIN_DIR}_logs"

export CUDA_VISIBLE_DEVICES='0'; \
  python3 ${CODE_PATH}/eval.py \
	--config_file=${CONFIG_PATH}/config.yaml \
	--config_name=eval_train \
	--train_dir=${TRAIN_DIR} \
  &>> "${LOGS_DIR}/log_eval_train.logs" &

export CUDA_VISIBLE_DEVICES='1'; \
  python3 ${CODE_PATH}/eval.py \
	--config_file=${CONFIG_PATH}/config.yaml \
	--config_name=eval_test \
	--train_dir=${TRAIN_DIR} \
  &>> "${LOGS_DIR}/log_eval_test.logs" &

wait

python3 ${CODE_PATH}/parse_events.py \
   --folder=${WORKDIR}/${TRAIN_DIR}

