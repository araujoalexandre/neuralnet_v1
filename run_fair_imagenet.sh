#!/usr/bash
DATASET_TO_USE=$1
PARTITION=$2

for DATASET in $DATASET_TO_USE
do
  # RUN NO NOISE
  PARAMS='{"reader": "'${DATASET}'", "train_with_noise": false, "distributions": false, "scale_noise": 0, "noise_in_eval": false}'
  ./sub/train.py config_template \
    --gpu_train '0,1,2,3' \
    --gpu_eval '' \
    --partition $PARTITION \
    --name "no_noise_$DATASET" \
    --params "${PARAMS}" | sbatch
  sleep 2
  # RUN WITH NOISE
  for DIST in 'l1' 'l2' 'exp' 'weibull'
  do

    # CHOICE OF NOISE VARIANCE
    if [ $DIST = 'l1' ]
    then
      # noise=('0.01' '0.05' '0.10' '0.15' '0.20' '0.25' '0.30' '0.35' '0.40' '0.45')
      noise=('0.01' '0.05'        '0.15'        '0.25'        '0.35'        '0.45')
    elif [ $DIST = 'l2' ]
    then
      # noise=('0.01' '0.23' '0.45' '0.68' '0.90' '1.00' '1.34' '1.55' '1.77' '2.00')
      noise=('0.01' '0.23'        '0.68'        '1.00'        '1.55'        '2.00')
    elif [ $DIST = 'exp' ]
    then
      # noise=('0.01' '0.08' '0.15' '0.23' '0.30' '0.38' '0.45' '0.53' '0.60' '0.68')
      noise=('0.01'        '0.15'        '0.30'        '0.45'        '0.60'       )
    elif [ $DIST = 'weibull' ]
    then
      # noise=('0.01' '0.23' '0.45' '0.68' '0.90' '1.00' '1.34' '1.55' '1.77' '2.00')
      noise=('0.01' '0.23'        '0.68'        '1.00'        '1.55'        '2.00')
    fi

    for NOISE_IN_EVAL in ${noise[@]};
    do
      PARAMS='{"reader": "'${DATASET}'", "train_with_noise": true, "distributions": "'${DIST}'", "scale_noise": '${NOISE_IN_EVAL}', "noise_in_eval": true}'
      ./sub/train.py config_template \
        --gpu_train '0,1,2,3,4,5,6,7' \
        --gpu_eval '' \
       	--partition $PARTITION \
        --name "xp_${DIST}_${DATASET}" \
        --params "${PARAMS}" | sbatch
      sleep 2
    done
  done
done
