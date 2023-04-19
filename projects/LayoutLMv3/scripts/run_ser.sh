config='/Users/wangnu/Documents/GitHub/mmocr/projects/LayoutLMv3/configs/ser/layoutlmv3_xfund_zh.py'

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

python tools/train.py \
${config} \
