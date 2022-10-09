_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_pipelines/satrn_pipeline.py',
    '../../_base_/recog_datasets/ST_MJ_train.py',
    '../../_base_/recog_datasets/academic_test.py',
    '../../_base_/schedules/schedule_adam_step_6e.py',
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

max_seq_len = 25

label_convertor = dict(
    type='AttnConvertor',
    dict_type='DICT90',
    with_unknown=True,
    max_seq_len=max_seq_len)

model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN', input_channels=3, hidden_dim=256),
    encoder=dict(
        type='SatrnEncoder',
        n_layers=6,
        n_head=8,
        d_k=256 // 8,
        d_v=256 // 8,
        d_model=256,
        n_position=100,
        d_inner=256 * 4,
        dropout=0.1),
    decoder=dict(
        type='NRTRDecoder',
        n_layers=6,
        d_embedding=256,
        n_head=8,
        d_model=256,
        d_inner=256 * 4,
        d_k=256 // 8,
        d_v=256 // 8),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len)

# optimizer
optimizer = dict(type='Adam', lr=3e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
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
