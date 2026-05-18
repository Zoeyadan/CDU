#!/bin/bash

# custom config
DATA=/mnt/data/.work/uda/Datasets # your directory

DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
NTOK=$5
DOMAINS=$6
GPU=$7
KD=$8

LOCATION=middle

TP=True
DEEPLAYER=None

TDEEP=False
VP=True
VDEEP=False
SHARE=True

DIR=output/cdu/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/deep${DEEP}_${LOCATION}_4/kd${KD}_${DOMAINS}_ntok${NTOK}

#if [ -d "$DIR" ]; then
#    echo "Results are available in ${DIR}, so skip this job"
#else
#    echo "Run this job and save the output to ${DIR}"

    python train.py \
        --gpu ${GPU} \
        --kd ${KD} \
        --backbone ${BACKBONE} \
        --domains ${DOMAINS} \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.CDU.STUDENT_N_CTX ${NTOK} \
        TRAINER.CDU.TEACHER_N_CTX ${NTOK} \
        TRAINER.CDU.LOCATION ${LOCATION}  \
        TRAINER.CDU.TP ${TP}\
        TRAINER.CDU.T_DEEP ${TDEEP} \
        TRAINER.CDU.VP ${VP} \
        TRAINER.CDU.V_DEEP ${VDEEP}\
        TRAINER.CDU.DEEP_LAYERS ${DEEPLAYER} \
        TRAINER.CDU.DEEP_SHARED ${SHARE}
#fi
