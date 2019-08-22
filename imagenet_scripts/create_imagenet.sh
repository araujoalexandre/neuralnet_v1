
#BSUB -J imagenet
#BSUB -gpu "num=4:mode=exclusive_process:mps=no:j_exclusive=yes"
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -W 1200
#BSUB -o /linkhome/rech/grpgen/ubx31mc/%J.train.out
#BSUB -e /linkhome/rech/grpgen/ubx31mc/%J.train.err

ml anaconda/py3 cudnn nccl
source activate tensorflow1.12-py3

PROJECTDIR='/linkhome/rech/grpgen/ubx31mc/neuralnet'
IMAGENET_DIR='/pwrwork/rech/jvd/ubx31mc/data/imagenet'
CODEDIR='/linkhome/rech/grpgen/ubx31mc/neuralnet/code/dataset'

# mkdir ${TMPDIR}/bounding_boxes
# 
# tar -C ${TMPDIR}/bounding_boxes -xvf /pwrdataset/ImageNet/ILSVRC2012_bbox_train_v2.tar.gz
# echo "done extract bounding boxes" >> build_imagenet.logs
# 
# python3 ${CODEDIR}/process_bounding_boxes.py ${TMPDIR}/bounding_boxes/ ${IMAGENET_DIR}/imagenet_lsvrc_2015_synsets.txt | sort > ${IMAGENET_DIR}/imagenet_2012_bounding_boxes.csv
# echo "done processing bounding boxes" >> build_imagenet.logs

echo ${TMPDIR} >> build_imagenet.logs

tar -C ${TMPDIR} -xf /pwrwork/rech/jvd/ubx31mc/data/imagenet/imagenet_object_localization.tar.gz
TRAIN_DIR=${TMPDIR}/ILSVRC/Data/CLS-LOC/train
VAL_DIR=${TMPDIR}/ILSVRC/Data/CLS-LOC/val

# Preprocess the validation data by moving the images into the appropriate
# sub-directory based on the label (synset) of the image.
VAL_LABELS_FILE=${IMAGENET_DIR}/imagenet_2012_validation_synset_labels.txt
python3 ${CODEDIR}/preprocess_imagenet_validation_data.py ${VAL_DIR} ${VAL_LABELS_FILE}

# convert JPEG to TFRecords
python3 ${CODEDIR}/build_imagenet.py \
  --train_directory ${TRAIN_DIR} \
  --validation_directory ${VAL_DIR} \
  --output_directory ${IMAGENET_DIR} \
  --num_threads 16 \
  --labels_file ${IMAGENET_DIR}/imagenet_lsvrc_2015_synsets.txt \
  --imagenet_metadata_file ${IMAGENET_DIR}/imagenet_metadata.txt \
  --bounding_box_file ${IMAGENET_DIR}/imagenet_2012_bounding_boxes.csv


