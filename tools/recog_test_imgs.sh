#!/bin/bash

DATE=`date +%Y-%m-%d`
TIME=`date +"%H-%M-%S"`

if [ $# -lt 5 ]
then
    echo "Usage: bash $0 CONFIG CHECKPOINT IMG_PREFIX IMG_LIST RESULTS_DIR"
    exit
fi

CONFIG_FILE=$1
CHECKPOINT=$2
IMG_ROOT_PATH=$3
IMG_LIST=$4
OUT_DIR=$5_${DATE}_${TIME}

mkdir ${OUT_DIR} -p &&

python tools/recog_test_imgs.py \
      --img_root_path ${IMG_ROOT_PATH} \
      --img_list ${IMG_LIST} \
      --config ${CONFIG_FILE} \
      --checkpoint ${CHECKPOINT} \
      --out_dir ${OUT_DIR}
