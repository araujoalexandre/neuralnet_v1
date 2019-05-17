#!/bin/bash
MODELS_FOLDER="models"
PARTITION="priority"
FOLDERS=$(find ${WORKDIR}/${MODELS_FOLDER}/* -type d | grep -v "logs" | sort)
for FOLDER in ${FOLDERS}
do
  FOLDER=$(basename $FOLDER)
  sed "s/PARTITION/${PARTITION}/g" compute_entropy.sh | \
    sed "s/FOLDER/${FOLDER}/g" | sbatch
done

MODELS_FOLDER="models_cifar"
PARTITION="learnfair"
FOLDERS=$(find ${WORKDIR}/${MODELS_FOLDER}/* -type d | grep -v "logs" | sort)
for FOLDER in ${FOLDERS}
do
  FOLDER=$(basename $FOLDER)
  sed "s/PARTITION/${PARTITION}/g" compute_entropy.sh | \
    sed "s/FOLDER/${FOLDER}/g" | sbatch
done
