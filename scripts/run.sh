
PROJECT_PATH="/linkhome/rech/grpgen/urz85ee/neuralnet"
CODE_PATH="${PROJECT_PATH}/code"
CONFIG_PATH=$1

DATE=$(date '+%Y-%m-%d_%H.%M.%S')
DATE=${DATE}.$(date +"%4N")
TRAIN_DIR="${WORKDIR}/models/${DATE}"
LOGS_DIR="${WORKDIR}/models/${DATE}_logs"
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

python3 ${CODE_PATH}/parse/parse_events.py \
        --folder=${TRAIN_DIR}

for attack in 'attack_fgsm' 'attack_pgd' 'attack_carlini' 
do
  export CUDA_VISIBLE_DEVICES='0'; 
  python3 ${CODE_PATH}/eval_under_attack.py \
          --config_file=${CONFIG_PATH} \
  	      --config_name=${attack} \
  	       --train_dir=${TRAIN_DIR} \
          &>> "${LOGS_DIR}/log_${attack}.logs" & 
  wait
done

