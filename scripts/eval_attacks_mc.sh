
PROJECT_PATH="/linkhome/rech/grpgen/urz85ee/neuralnet"
CODE_PATH="${PROJECT_PATH}/code"
MODELS_DIR="/pwrproj/rech/zbe/urz85ee/models"

DATE=$1
SAMPLE=$2
TRAIN_DIR="${MODELS_DIR}/${DATE}"
LOGS_DIR="${MODELS_DIR}/${DATE}_logs"
CONFIG_PATH="${LOGS_DIR}/model_flags.yaml"

fgm="attack_fgsm_mc_${SAMPLE}"
pgd="attack_pgd_mc_${SAMPLE}"
carlini="attack_carlini_mc_${SAMPLE}"

for attack in ${fgm} ${pgd} ${carlini}
do
  export CUDA_VISIBLE_DEVICES='0'; 
  python3 ${CODE_PATH}/eval_under_attack.py \
          --config_file=${CONFIG_PATH} \
  	      --config_name=${attack} \
  	      --train_dir=${TRAIN_DIR} \
          &>> "${LOGS_DIR}/log_${attack}.logs" & 
  wait
done

