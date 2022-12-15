_base_ = [
    'sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real.py',
]

model = dict(decoder=dict(type='SequentialSARDecoder'))
