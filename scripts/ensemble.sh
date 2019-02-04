
WORKDIR="/pwrwork/rech/jvd/ubx31mc/neuralnet/mnist_models"
PROJECT_PATH="/linkhome/rech/grpgen/ubx31mc/neuralnet"
CODE_PATH="${PROJECT_PATH}/code"
CONFIG_PATH="${PROJECT_PATH}/config"
CONFIG_NAME="config_mnist.yaml"

for i in {1..25}
do
  
  DATE=$(date '+%Y-%m-%d_%H.%M.%S')
  TRAIN_DIR="${WORKDIR}/${DATE}"
  LOGS_DIR="${WORKDIR}/${DATE}_logs"
  mkdir ${LOGS_DIR}
  cp ${CONFIG_PATH}/${CONFIG_NAME} ${LOGS_DIR}/model_flags.yaml
  
  python3 ${CODE_PATH}/train.py \
    --config_file=${CONFIG_PATH}/${CONFIG_NAME} \
    --config_name=train \
    --train_dir=${TRAIN_DIR} \
    &>> "${LOGS_DIR}/log_train.logs" &

  wait

done
