#!/usr/bash
DATASET_TO_USE=$1
PARTITION=$2

for DATASET in $DATASET_TO_USE
do
  # RUN NO NOISE
  PARAMS='{"reader": "'${DATASET}'", "train_with_noise": false, "distributions": false, "scale_noise": 0, "noise_in_eval": false}'
  ./sub/train.py config_img_noise \
    --gpu_train '0,1,2,3' \
    --gpu_eval '' \
    --partition $PARTITION \
    --name "no_noise_$DATASET" \
    --params "${PARAMS}" | sbatch
  sleep 2
  # RUN WITH NOISE
  for DIST in 'l1' 'l2' 'exp'
  do

    # LIST OF NOISE VARIANCE
    noise=('0.01' '0.015' '0.02' '0.03' '0.05' '0.08' '0.13' '0.20' '0.32' '0.5') 

    for NOISE_IN_EVAL in ${noise[@]};
    do
      PARAMS='{"reader": "'${DATASET}'", "train_with_noise": true, "distributions": "'${DIST}'", "scale_noise": '${NOISE_IN_EVAL}', "noise_in_eval": true}'
      ./sub/train.py config_img_noise \
        --gpu_train '0,1,2,3' \
        --gpu_eval '' \
       	--partition $PARTITION \
        --name "xp_${DIST}_${DATASET}" \
        --params "${PARAMS}" | sbatch
      sleep 2
    done
  done
done



