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

DIR=output/cdusource/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/deep${DEEP}_${LOCATION}/${DOMAINS}_ntok${NTOK}

#if [ -d "$DIR" ]; then
#    echo "Results are available in ${DIR}, so skip this job"
#else
#    echo "Run this job and save the output to ${DIR}"

    python train.py \
        --gpu ${GPU} \
        --backbone ${BACKBONE} \
        --domains ${DOMAINS} \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.CDUSOURCE.NUM_TOKENS ${NTOK} \
        TRAINER.CDUSOURCE.N_CTX ${NTOK} \
        TRAINER.CDUSOURCE.T_DEEP ${DEEP} \
        TRAINER.CDUSOURCE.V_DEEP ${DEEP} \
        TRAINER.CDUSOURCE.LOCATION ${LOCATION} \
        TRAINER.CDUSOURCE.DEEP_LAYERS ${DEEPLAYER}
#fi
