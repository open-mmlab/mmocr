PROJ_ROOT=$(pwd)
DATASET_ZOO_PATH=${PROJ_ROOT}/dataset_zoo
NPROC=8
TASKS=('ser' 're')
SPLITS=('train' 'test')
# DATASET_NAME=('xfund/de' 'xfund/es' 'xfund/fr' 'xfund/jt' 'xfund/ja' 'xfund/pt' 'xfund/zh')
DATASET_NAME=('xfund/zh')

for TASK in ${TASKS[@]}
do
    python tools/dataset_converters/prepare_dataset.py \
    ${DATASET_NAME[@]} \
    --nproc ${NPROC} \
    --task ${TASK} \
    --splits ${SPLITS[@]} \
    --dataset-zoo-path ${DATASET_ZOO_PATH} \
    --overwrite-cfg
done
