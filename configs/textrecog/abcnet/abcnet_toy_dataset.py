_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_pipelines/abinet_pipeline.py',
    '../../_base_/recog_datasets/toy_data.py',
    '../../_base_/schedules/schedule_adam_step_20e.py',
]

max_seq_len = 10
num_chars = 39

label_convertor = dict(
    type='AttnConvertor',
    dict_type='DICT36',
    with_unknown=True,
    lower=True,
    max_seq_len=max_seq_len)

model = dict(
    type='ABCRecognizer',
    preprocessor=None,
    backbone=dict(type='ResNetABI'),
    encoder=dict(type='ABCRecogEncoder', num_channels=512),
    decoder=dict(type='ABCRecogDecoder', in_channels=512, num_chars=num_chars),
    loss=dict(type='NLLLoss', ignore_first_char=True),
    max_seq_len=max_seq_len,
    label_convertor=label_convertor,
    pretrained=None)

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')

cudnn_benchmark = True
