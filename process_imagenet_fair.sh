#!/bin/bash
#SBATCH --job-name=imagenet
#SBATCH --output=/private/home/laurentmeunier/alex/neuralnet/sample-%j.out
#SBATCH --error=/private/home/laurentmeunier/alex/neuralnet/sample-%j.err
#SBATCH --time=4300
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --get-user-env

IMAGENET_FILES='${PROJECTDIR}/code/dataset/ImageNet_Files'
OUTPUT_DIR='${WORKDIR}/data/imagenet'
mkdir ${OUTPUT_DIR}

TRAIN_DIR=/datasets01/imagenet_full_size/061417/train
VAL_DIR=/datasets01/imagenet_full_size/061417/val

# convert JPEG to TFRecords
srun -o /private/home/laurentmeunier/alex/neuralnet/build_imagenet.logs \
  python3 ${PROJECTDIR}/code/dataset/build_imagenet.py \
  --train_directory ${TRAIN_DIR} \
  --validation_directory ${VAL_DIR} \
  --output_directory ${OUTPUT_DIR} \
  --num_threads 32 \
  --labels_file ${IMAGENET_FILES}/imagenet_lsvrc_2015_synsets.txt \
  --imagenet_metadata_file ${IMAGENET_FILES}/imagenet_metadata.txt \
  --bounding_box_file ${IMAGENET_FILES}/imagenet_2012_bounding_boxes.csv
