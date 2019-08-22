#!/usr/bin/bash
ID=$1
TYPE=$2
CONFIG=$3
FOLDER_ID=$4

ml anaconda/py3 cudnn nccl
source activate tensorflow1.12-py3
export LD_LIBRARY_PATH="/pwrlocal/pub/nccl/2.2.13-1/lib:/pwrlocal/pub/cudnn/7.1.4/lib:/pwrlocal/pub/openmpi/2.1.2/arch/gcc-4.8/lib:/pwrlocal/pub/cudnn/7.1.4/lib64:/pwrlocal/lsf/10.1/linux3.10-glibc2.17-ppc64le/lib"
export TF_CONFIG=${CONFIG}

CONFIG_PATH="${PROJECTDIR}/config_gen/config_imagenet_0.yaml"
TRAIN_DIR="${WORKDIR}/models/${FOLDER_ID}"
LOGS_DIR="${WORKDIR}/models/${FOLDER_ID}_logs"
DATA_DIR="${WORKDIR}/data"

if [ $ID -eq 0 ]
then
  mkdir ${LOGS_DIR}
  cp ${CONFIG_PATH} ${LOGS_DIR}/model_flags.yaml
fi

if [ $TYPE = 'ps' ]
then
  LOGS_NAME=$LOGS_DIR"/log_train_ps_${ID}.logs"
  export CUDA_VISIBLE_DEVICES='';
elif [ $TYPE = 'master' ]
then
  export CUDA_VISIBLE_DEVICES='0,1,2,3';
  LOGS_NAME=$LOGS_DIR"/log_train_master.logs"
elif [ $TYPE = 'worker' ]
then
  export CUDA_VISIBLE_DEVICES='0,1,2,3';
  LOGS_NAME=$LOGS_DIR"/log_train_worker_${ID}.logs"
fi

python3 ${PROJECTDIR}/code/train.py \
  --config_file=${CONFIG_PATH} \
  --config_name=train \
  --train_dir=${TRAIN_DIR} \
  --data_dir=${DATADIR} \
  &>> ${LOGS_NAME} &



