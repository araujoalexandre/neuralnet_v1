#BSUB -J train
#BSUB -gpu "num=4:mode=exclusive_process:mps=no:j_exclusive=yes"
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -W 1000
#BSUB -o /linkhome/rech/grpgen/ubx31mc/neuralnet/job.%J.train.out
#BSUB -e /linkhome/rech/grpgen/ubx31mc/neuralnet/job.%J.train.err

ml anaconda/py3 cudnn nccl
source activate tensorflow1.12-py3
export LD_LIBRARY_PATH="/pwrlocal/pub/nccl/2.2.13-1/lib:/pwrlocal/pub/cudnn/7.1.4/lib:/pwrlocal/pub/openmpi/2.1.2/arch/gcc-4.8/lib:/pwrlocal/pub/cudnn/7.1.4/lib64:/pwrlocal/lsf/10.1/linux3.10-glibc2.17-ppc64le/lib"

FOLDER_ID=$(date '+%Y-%m-%d_%H.%M.%S')
FOLDER_ID=${FOLDER_ID}.$(date +"%4N")

CONFIG_PATH="${PROJECTDIR}/config_gen/config_imagenet_0.yaml"
TRAIN_DIR="${WORKDIR}/models/${FOLDER_ID}"
LOGS_DIR="${WORKDIR}/models/${FOLDER_ID}_logs"
DATA_DIR="${WORKDIR}/data"

mkdir ${LOGS_DIR}
cp ${CONFIG_PATH} ${LOGS_DIR}/model_flags.yaml

export CUDA_VISIBLE_DEVICES='0,1,2,3';
LOGS_NAME=$LOGS_DIR"/log_train.logs"

python3 ${PROJECTDIR}/code/train.py \
  --config_file=${CONFIG_PATH} \
  --config_name=train \
  --train_dir=${TRAIN_DIR} \
  --data_dir=${DATADIR} \
  &>> ${LOGS_NAME} &

# TF_SLIM='/linkhome/rech/grpgen/ubx31mc/tf_models/research/slim'
# python3 ${TF_SLIM}/train_image_classifier.py \
#   --train_dir=${TRAIN_DIR} \
#   --dataset_name=imagenet \
#   --dataset_split_name=train \
#   --dataset_dir=${WORKDIR}/data/imagenet \
#   --model_name=inception_resnet_v2 \
#   --batch_size 32 \
#   &>> ${LOGS_NAME} &







