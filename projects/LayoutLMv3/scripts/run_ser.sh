config='/Users/wangnu/Documents/GitHub/mmocr/projects/LayoutLMv3/configs/ser/layoutlmv3_1k_xfund_zh_1xbs8.py'

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH='/Users/wangnu/Documents/GitHub/mmocr'

python tools/train.py \
${config} \
