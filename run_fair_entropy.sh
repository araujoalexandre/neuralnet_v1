#!/usr/bin/bash
MODELS_FOLDER=$1
PARTITION=$2
FOLDERS=$(find ${WORKDIR}/${MODELS_FOLDER}/* -type d | grep -v "logs" | sort)
for FOLDER in ${FOLDERS}
do
  sed "s/PARTITION/${PARTITION}/g" compute_entropy.sh | \
    sed "s/FOLDER/${MODELS_FOLDER}/g" | sbatch 
done


