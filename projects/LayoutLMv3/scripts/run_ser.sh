config='/Users/wangnu/Documents/GitHub/mmocr/projects/LayoutLMv3/configs/ser/layoutlmv3_xfund_zh.py'

export TOKENIZERS_PARALLELISM=false

python tools/train.py \
${config} \
