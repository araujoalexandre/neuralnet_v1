#!/bin/bash
PARTITION="priority"

# cherry pick of randomized models
FOLDERS=('2019-05-15_08.14.10_1357' '2019-05-15_08.13.47_7189' '2019-05-15_08.14.12_2848' '2019-05-15_08.13.50_2361')

for FOLDER in ${FOLDERS[@]}
do
  
  PARAMS='{"eval_num_gpu": 1, "eval_batch_size": 2, "attack_sample": 80, "data_pattern": "test-00001*,test-00002*", "sample": 80, "CarliniWagnerL2": {"max_iterations": 30}}'
  
  # generate the sbatch and run on cluster
  ${PROJECTDIR}/sub/attacks.py ${FOLDER} \
    --attack 'carlini' \
    --gpu '0' \
    --partition ${PARTITION} \
    --params "${PARAMS}" | sbatch

done
