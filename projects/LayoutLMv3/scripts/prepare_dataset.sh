PROJ_ROOT=$(pwd)
DATASET_ZOO_PATH=${PROJ_ROOT}/dataset_zoo
NPROC=8
TASKS=('ser' 're')
# DATASET_NAME=('xfund/de' 'xfund/es' 'xfund/fr' 'xfund/jt' 'xfund/ja' 'xfund/pt' 'xfund/zh')
DATASET_NAME=('xfund/zh')

for TASK in ${TASKS[@]}
do
    python tools/dataset_converters/prepare_dataset.py \
    ${DATASET_NAME[@]} \
    --nproc ${NPROC} \
    --task ${TASK} \
    --splits train test \
    --overwrite-cfg \
    --dataset-zoo-path ${DATASET_ZOO_PATH}
done
