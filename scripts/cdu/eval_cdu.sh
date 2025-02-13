#!/bin/bash

# custom config
DATA=/Workplace_sdb/dxw/data # your directory

DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
NTOK=$5
DOMAINS=$6
GPU=$7

LOCATION=middle
DEEP=False
DEEPLAYER=None
SHOTS=0

TEST_BATCH_SIZE=1

DIR=output/promptkd/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/deep${DEEP}_${LOCATION}_origin/kd100_${DOMAINS}_ntok${NTOK}

python train.py \
    --gpu ${GPU} \
    --backbone ${BACKBONE} \
    --domains ${DOMAINS} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${DIR} \
    --eval-only \
    TRAINER.CDUTARGET.NUM_TOKENS ${NTOK} \
    TRAINER.CDUTARGET.N_CTX ${NTOK} \
    TRAINER.CDUTARGET.T_DEEP ${DEEP} \
    TRAINER.CDUTARGET.V_DEEP ${DEEP} \
    TRAINER.CDUTARGET.LOCATION ${LOCATION} \
    TRAINER.CDUTARGET.DEEP_LAYERS ${DEEPLAYER} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATALOADER.TEST.BATCH_SIZE ${TEST_BATCH_SIZE}

