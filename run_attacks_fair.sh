#!/bin/bash
ATTACK=$1
PARTITION=$2

# cherry pick of randomized models 
FOLDERS=('2019-05-07_08.35.54_1410' '2019-05-07_08.35.11_7786' '2019-05-07_08.36.18_6733' '2019-05-07_08.36.58_6857' '2019-05-08_02.33.57_3626' '2019-05-08_02.33.31_2595' '2019-05-08_02.34.23_2430' '2019-05-08_02.34.48_1401')

for FOLDER in ${FOLDERS}
do

  # define the itertions array given the attack
  if [ ${ATTACK} = "carlini" ] || [ ${ATTACK} = "elasticnet" ]
  then
    ITERATIONS=('20' '50' '60')
  elif [ ${ATTACK} = "pgd" ]
  then
    ITERATIONS=('10' '20' '30')
  fi

  for ITER in ${ITERATIONS[@]}
  do
   
    # create the params string from iter parameter
    if [ ${ATTACK} = "carlini" ]
    then
      PARAMS='{"eval_batch_size": 4, "attack_sample": 80, "data_pattern": "test-00001*,test-00002*", "sample": 80, "CarliniWagnerL2": {"max_iterations": '${ITER}'}}'
    elif [ ${ATTACK} = "pgd" ]
    then
      PARAMS='{"eval_batch_size": 4, "attack_sample": 80, "data_pattern": "test-00001*,test-00002*", "sample": 80, "ProjectedGradientDescent": {"nb_iter": '${ITER}'}}'
    elif [ ${ATTACK} = "elasticnet" ]
    then
      PARAMS='{"eval_batch_size": 4, "attack_sample": 80, "data_pattern": "test-00001*,test-00002*", "sample": 80, "ElasticNet": {"max_iterations": '${ITER}'}}'
    fi
    echo ${PARAMS}

    # generate the sbatch and run on cluster
    ${PROJECTDIR}/sub/attacks.py ${FOLDER} \
      --attack ${ATTACK} \
      --gpu '0' \
      --partition ${PARTITION} \
      --params "${PARAMS}" | sbatch

  done

FOLDERS=('2019-05-07_08.34.56_0963' '2019-05-08_02.33.24_9114')
for FOLDER in ${FOLDERS}
do

  # define the itertions array given the attack
  if [ ${ATTACK} = "carlini" ] || [ ${ATTACK} = "elasticnet" ]
  then
    ITERATIONS=('20' '50' '60')
  elif [ ${ATTACK} = "pgd" ]
  then
    ITERATIONS=('10' '20' '30')
  fi

  for ITER in ${ITERATIONS[@]}
  do
   
    # create the params string from iter parameter
    if [ ${ATTACK} = "carlini" ]
    then
      PARAMS='{"eval_batch_size": 400, "attack_sample": 1, "data_pattern": "test-00001*,test-00002*", "CarliniWagnerL2": {"max_iterations": '${ITER}'}}'
    elif [ ${ATTACK} = "pgd" ]
    then
      PARAMS='{"eval_batch_size": 400, "attack_sample": 1, "data_pattern": "test-00001*,test-00002*", "ProjectedGradientDescent": {"nb_iter": '${ITER}'}}'
    elif [ ${ATTACK} = "elasticnet" ]
    then
      PARAMS='{"eval_batch_size": 400, "attack_sample": 1, "data_pattern": "test-00001*,test-00002*", "ElasticNet": {"max_iterations": '${ITER}'}}'
    fi
    echo ${PARAMS}

    # generate the sbatch and run on cluster
    ${PROJECTDIR}/sub/attacks.py ${FOLDER} \
      --attack ${ATTACK} \
      --gpu '0' \
      --partition ${PARTITION} \
      --params "${PARAMS}" | sbatch

  done
done





