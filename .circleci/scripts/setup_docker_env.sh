#!/bin/bash

TORCH=$1
CUDA=$2
CUDNN=$3

# 10.2 -> cu102
MMCV_CUDA="cu`echo ${CUDA} | tr -d '.'`"

TORCH_VER_ARR=(${TORCH//./ })
if [ ${TORCH_VER_ARR[2]} -eq 0 ]
then
    MMCV_TORCH=${TORCH}
else
    # MMCV only provides pre-compiled packages for torch 1.x.0
    # which works for any subversions of torch 1.x.
    # We force the torch version to be 1.x.0 to ease package searching
    # and avoid unnecessary rebuild during MMCV's installation.
    TORCH_VER_ARR[2]=0
    printf -v MMCV_TORCH "%s." "${TORCH_VER_ARR[@]}"
    # Remove the last dot
    MMCV_TORCH=${MMCV_TORCH%?}
fi

echo "Build Docker image"
docker build .circleci/docker -t mmocr:gpu --build-arg PYTORCH=${TORCH} --build-arg CUDA=${CUDA} --build-arg CUDNN=${CUDNN}
docker run --gpus all -t -d -v /home/circleci/project:/mmocr -w /mmocr --name mmocr mmocr:gpu

echo "Install mmocr dependencies"
docker exec mmocr pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/${MMCV_CUDA}/torch${MMCV_TORCH}/index.html
docker exec mmocr pip install mmdet
docker exec mmocr pip install -r requirements.txt
