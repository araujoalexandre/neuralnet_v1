#! /bin/bash

ATTACK_TYPE=$1
FOLDERS=$(find ${2}/* -type d | grep -vE "logs")
CONFIG_DIR="/linkhome/rech/grpgen/ubx31mc/neuralnet/config_attacks/${ATTACK_TYPE}"

if [ $ATTACK_TYPE = "fgm_l1" ] || [ $ATTACK_TYPE = "fgm_l2" ] || [ $ATTACK_TYPE = "fgm_inf" ];
then
  code_file="eval_under_attack_v2.py"

elif [ $ATTACK_TYPE = "carlini" ] || [ $ATTACK_TYPE = "deepfool" ];
then
  code_file="eval_under_attack.py"

else
  echo "attack type wrong !"
fi


for TRAIN_DIR in ${FOLDERS}
do
  LOGS_DIR="${TRAIN_DIR}_logs"
  DATE_LOG=$(basename ${LOGS_DIR})

  python3 code/${code_file} \
    --config_file="${LOGS_DIR}/model_flags.yaml" \
    --config_name=attack \
    --train_dir=${TRAIN_DIR} \
    &>> ${LOGS_DIR}/log_${ATTACK_TYPE}.logs &
  wait
done
