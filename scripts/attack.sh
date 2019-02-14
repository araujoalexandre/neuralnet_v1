
FOLDER="/srv/osirim/aaraujo/neuralnet/models"
PROJECT_PATH="/home/aaraujo/neuralnet"
CODE_PATH="${PROJECT_PATH}/code"
DATE="$1"
TRAIN_DIR="${FOLDER}/${DATE}"
LOGS_DIR="${TRAIN_DIR}_logs"
CONFIG_FILE="${LOGS_DIR}/model_flags.yaml"

for attack in 'attack_fgsm' 'attack_pgd' 'attack_carlini' 
do
  export CUDA_VISIBLE_DEVICES='';
  python3 ${CODE_PATH}/eval_under_attack.py \
          --config_file=${CONFIG_FILE} \
  	  --config_name=${attack} \
  	  --train_dir=${TRAIN_DIR} \
          &>> "${LOGS_DIR}/log_${attack}.logs" & 
  wait
done
