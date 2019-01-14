#! /bin/bash

attack_type=$1
dataset=$2
noise=$3

if [ $attack_type = "fgm" ];
then

  declare -a ARRAY=( 
    "fgm_${noise}_L1_0.3"
    "fgm_${noise}_L2_0.3"
    "fgm_${noise}_inf_0.03"
    "fgm_${noise}_inf_0.3"
   )
  code_file="eval_under_attack_v2.py"

elif [ $attack_type = "carlini" ];
then
 
  declare -a ARRAY=( 
    "carlini_${noise}"
   )
  code_file="eval_under_attack.py"

elif [ $attack_type = "deepfool" ];
then

  declare -a ARRAY=( 
    "deepfool_${noise}"
   )
  code_file="eval_under_attack.py"

else
  echo "attack type wrong !"
fi


for config in ${ARRAY[@]}
do
  export CUDA_VISIBLE_DEVICES='';
  python3 code/${code_file} \
    --config_file=config_attacks/${dataset}/${config}.yaml \
    --config_name=attack \
    &>> log_${dataset}_${config}.logs &
  wait
done
