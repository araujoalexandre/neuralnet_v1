
MODELS_PATH="/home/aaraujo/neuralnet_structured_matrices/models"
TRAIN_DIR=$(date '+%Y-%m-%d_%H.%M.%S')

DIR_LOGS="${MODELS_PATH}/${TRAIN_DIR}_logs"
mkdir ${DIR_LOGS}
cp ./config.yaml ${DIR_LOGS}/model_flags.yaml

nohup python3 code/train.py \
	--config_file=config.yaml \
	--config_name=train \
	--train_dir=${TRAIN_DIR} \
  &>> "${DIR_LOGS}/log_train.logs" &


nohup python3 code/eval.py \
	--config_file=config.yaml \
	--config_name=eval \
  &>> "${DIR_LOGS}/log_eval.logs" &

