
WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/cifar10_circulant"

PROJECT_PATH="/linkhome/rech/grpgen/ubx31mc/neuralnet"
CONFIG_PATH="${PROJECT_PATH}/config/config.yaml"
CODE_PATH="${PROJECT_PATH}/code"

if [ "$#" -ne 1 ]
then
  DATE=$(date '+%Y-%m-%d_%H.%M.%S')
  TRAIN_DIR="${WORKDIR}/${DATE}"
  LOGS_DIR="${WORKDIR}/${DATE}_logs"
  mkdir ${LOGS_DIR}
  cp ${CONFIG_PATH} ${LOGS_DIR}/model_flags.yaml
else
  DATE=$1
  TRAIN_DIR="${WORKDIR}/${DATE}"
  LOGS_DIR="${WORKDIR}/${DATE}_logs"
fi

# export CUDA_VISIBLE_DEVICES='0,1';
python3 ${CODE_PATH}/train.py \
	--config_file=${CONFIG_PATH} \
	--config_name=train \
	--train_dir=${TRAIN_DIR} \
  &>> "${LOGS_DIR}/log_train.logs" &

# export CUDA_VISIBLE_DEVICES='2,3';
python3 ${CODE_PATH}/eval.py \
	--config_file=${CONFIG_PATH} \
	--config_name=eval_test \
	--train_dir=${TRAIN_DIR} \
  &>> "${LOGS_DIR}/log_eval_test.logs" &
wait

python3 ${CODE_PATH}/parse_events.py \
   --folder=${TRAIN_DIR}

