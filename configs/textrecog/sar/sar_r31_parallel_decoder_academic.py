_base_ = [
    'sar.py',
    '../../_base_/recog_datasets/ST_SA_MJ_real_train.py',
    '../../_base_/recog_datasets/academic_test.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_5e.py',
]

# dataset settings
file_client_args = dict(backend='disk')
default_hooks = dict(logger=dict(type='LoggerHook', interval=100))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=4),
    dict(type='PadToWidth', width=160),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=4),
    dict(type='PadToWidth', width=160),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio',
                   'instances'))
]

# dataset settings
ic11_rec_train = _base_.ic11_rec_train
ic13_rec_train = _base_.ic13_rec_train
ic15_rec_train = _base_.ic15_rec_train
cocov1_rec_train = _base_.cocov1_rec_train
iiit5k_rec_train = _base_.iiit5k_rec_train
st_add_rec_train = _base_.st_add_rec_train
st_rec_train = _base_.st_rec_train
mj_rec_trian = _base_.mj_rec_trian

ic11_rec_train.pipeline = test_pipeline
ic13_rec_train.pipeline = test_pipeline
ic15_rec_train.pipeline = test_pipeline
cocov1_rec_train.pipeline = test_pipeline
iiit5k_rec_train.pipeline = test_pipeline
st_add_rec_train.pipeline = test_pipeline
st_rec_train.pipeline = test_pipeline
mj_rec_trian.pipeline = test_pipeline
repeat_ic11 = dict(type='RepeatDataset', dataset=ic11_rec_train, times=20)
repeat_ic13 = dict(type='RepeatDataset', dataset=ic13_rec_train, times=20)
repeat_ic15 = dict(type='RepeatDataset', dataset=ic15_rec_train, times=20)
repeat_cocov1 = dict(type='RepeatDataset', dataset=cocov1_rec_train, times=20)
repeat_iiit5k = dict(type='RepeatDataset', dataset=iiit5k_rec_train, times=20)

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            repeat_ic11, repeat_ic13, repeat_ic15, repeat_cocov1,
            repeat_iiit5k, st_add_rec_train, st_rec_train, mj_rec_trian
        ]))

test_cfg = dict(type='MultiTestLoop')
val_cfg = dict(type='MultiValLoop')
val_dataloader = _base_.val_dataloader
test_dataloader = _base_.test_dataloader
for dataloader in test_dataloader:
    dataloader['dataset']['pipeline'] = test_pipeline
for dataloader in val_dataloader:
    dataloader['dataset']['pipeline'] = test_pipeline
visualizer = dict(type='TextRecogLocalVisualizer', name='visualizer')
