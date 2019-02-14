
PROJECT_PATH="/home/aaraujo/neuralnet"
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

# export CUDA_VISIBLE_DEVICES='2,3';
export CUDA_VISIBLE_DEVICES='';
python3 ${CODE_PATH}/eval.py \
	--config_file=${CONFIG_PATH} \
	--config_name=eval_test \
	--train_dir=${TRAIN_DIR} \
  &>> "${LOGS_DIR}/log_eval_test.logs" &

wait

python3 ${CODE_PATH}/parse/parse_events.py \
        --folder=${TRAIN_DIR}

for attack in [ 'attack_fgsm' 'attack_pgd' 'attack_carlini' ]
do
  python3 ${CODE_PATH}/eval_under_attack.py \
         	--config_file=${CONFIG_FILE} \
  	      --config_name=${attack} \
  	      --train_dir=${TRAIN_DIR} \
          &>> "${LOGS_DIR}/log_${attack}.logs" & 
  wait
done


