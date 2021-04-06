#!/bin/bash

DATE=`date +%Y-%m-%d`
TIME=`date +"%H-%M-%S"`

if [ $# -lt 3 ]
then
    echo "Usage: bash $0 CONFIG CHECKPOINT SHOW_DIR"
    exit
fi

CONFIG_FILE=$1
CHECKPOINT=$2
SHOW_DIR=$3_${DATE}_${TIME}

mkdir ${SHOW_DIR} -p &&

python tools/kie_test_imgs.py \
      ${CONFIG_FILE} \
      ${CHECKPOINT} \
      --show-dir ${SHOW_DIR}
