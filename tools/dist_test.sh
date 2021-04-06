#!/usr/bin/env bash

if [ $# -lt 3 ]
then
    echo "Usage: bash $0 CONFIG CHECKPOINT GPUS"
    exit
fi

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
