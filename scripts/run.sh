# WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/cifar10_test_with_random"
WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/cifar10_circulant"
# WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/cifar10_acdc"
# WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/cifar10_tensortrain"
# WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/cifar10_lowrank"
# WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/youtube_models"
# WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/youtube_with_tensortrain"
# WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/youtube_lowrank"
# WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/cifar10_circulant_relu"
# WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/cifar10_toeplitz"

PROJECT_PATH="/linkhome/rech/grpgen/ubx31mc/neuralnet"
CODE_PATH="${PROJECT_PATH}/code"
CONFIG_NAME=$1
CONFIG_PATH="${PROJECT_PATH}/config/${CONFIG_NAME}.yaml"

DATE=$(date '+%Y-%m-%d_%H.%M.%S')
TRAIN_DIR="${WORKDIR}/${DATE}"
LOGS_DIR="${WORKDIR}/${DATE}_logs"
mkdir ${LOGS_DIR}
cp ${CONFIG_PATH} ${LOGS_DIR}/model_flags.yaml

export CUDA_VISIBLE_DEVICES='0,1';
python3 ${CODE_PATH}/train.py \
	--config_file=${CONFIG_PATH} \
	--config_name=train \
	--train_dir=${TRAIN_DIR} \
  &>> "${LOGS_DIR}/log_train.logs" &

export CUDA_VISIBLE_DEVICES='2,3';
python3 ${CODE_PATH}/eval.py \
	--config_file=${CONFIG_PATH} \
	--config_name=eval_test \
	--train_dir=${TRAIN_DIR} \
  &>> "${LOGS_DIR}/log_eval_test.logs" &
wait

python3 ${CODE_PATH}/parse_events.py \
   --folder=${TRAIN_DIR}

