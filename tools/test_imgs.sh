#!/bin/bash

DATE=`date +%Y-%m-%d`
TIME=`date +"%H-%M-%S"`

if [ $# -lt 5 ]
then
    echo "Usage: bash $0 CONFIG CHECKPOINT IMG_ROOT_PATH IMG_LIST OUT_DIR"
    exit
fi

CONFIG_FILE=$1
CHECKPOINT=$2
IMG_ROOT_PATH=$3
IMG_LIST=$4
OUT_DIR=$5

mkdir ${OUT_DIR} -p &&


python tools/test_imgs.py \
     ${CONFIG_FILE} ${CHECKPOINT} ${IMG_ROOT_PATH} ${IMG_LIST} \
      --out-dir ${OUT_DIR}
