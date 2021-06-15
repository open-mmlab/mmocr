#!/usr/bin/env bash

set -x
export PYTHONPATH=`pwd`:$PYTHONPATH

if [ $# -lt 5 ]
then
    echo "Usage: bash $0 CONFIG CHECKPOINT IMG_PREFIX IMG_LIST RESULTS_DIR"
    exit
fi

CONFIG_FILE=$1
CHECKPOINT=$2
IMG_ROOT_PATH=$3
IMG_LIST=$4
OUT_DIR=$5
PY_ARGS=${@:6}

mkdir ${OUT_DIR} -p &&

python tools/recog_test_imgs.py \
      ${IMG_ROOT_PATH} \
      ${IMG_LIST} \
      ${CONFIG_FILE} \
      ${CHECKPOINT} \
      --out_dir ${OUT_DIR} ${PY_ARGS}
