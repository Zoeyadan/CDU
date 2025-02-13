#!/bin/bash

# custom config
DATA=/Workplace_sdb/dxw/data  # your directory

DATASET=$1 # name of the dataset
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
DOMAINS=$5
GPU=$6
SHOTS=0

OUTPUT_DIR=output/clip/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAINS}
DIR=/Workpalce_sdc/dxw/PDA/assets

python train.py \
    --gpu ${GPU} \
    --backbone ${BACKBONE} \
    --domains ${DOMAINS} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CLIP/${CFG}.yaml \
    --output-dir ${OUTPUT_DIR} \
    --model-dir ${DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} 
