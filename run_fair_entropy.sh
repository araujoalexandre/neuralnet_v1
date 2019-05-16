#!/bin/bash
MODELS_FOLDER=$1
PARTITION=$2
FOLDERS=$(find ${WORKDIR}/${MODELS_FOLDER}/* -type d | grep -v "logs" | sort)
for FOLDER in ${FOLDERS}
do
  FOLDER=$(basename $FOLDER)
  sed "s/PARTITION/${PARTITION}/g" compute_entropy.sh | \
    sed "s/FOLDER/${FOLDER}/g" | sbatch
done


