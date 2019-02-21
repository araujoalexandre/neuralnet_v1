
PROJECT_PATH="/linkhome/rech/grpgen/urz85ee/neuralnet"
CODE_PATH="${PROJECT_PATH}/code"
MODELS_DIR="/pwrproj/rech/zbe/urz85ee/models"

DATE=$1
TRAIN_DIR="${MODELS_DIR}/${DATE}"
LOGS_DIR="${MODELS_DIR}/${DATE}_logs"
CONFIG_PATH="${LOGS_DIR}/model_flags.yaml"

# export CUDA_VISIBLE_DEVICES='0,1';
# python3 ${CODE_PATH}/eval.py \
#       --config_file=${CONFIG_PATH} \
#       --config_name=eval_test \
#       --train_dir=${TRAIN_DIR} \
#       &>> "${LOGS_DIR}/log_eval_test.logs" &
# 
# wait

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

