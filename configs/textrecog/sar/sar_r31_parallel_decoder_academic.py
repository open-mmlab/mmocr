_base_ = [
    '../../_base_/recog_datasets/mjsynth.py',
    '../../_base_/recog_datasets/synthtext.py',
    '../../_base_/recog_datasets/coco_text_v1.py'
    '../../_base_/recog_datasets/cute80.py',
    '../../_base_/recog_datasets/iiit5k.py',
    '../../_base_/recog_datasets/svt.py',
    '../../_base_/recog_datasets/svtp.py',
    '../../_base_/recog_datasets/icdar2013.py',
    '../../_base_/recog_datasets/icdar2015.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_5e.py',
    'sar.py',
]

file_client_args = dict(backend='disk')
default_hooks = dict(logger=dict(type='LoggerHook', interval=100))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ignore_empty=True,
        min_size=5),
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
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

# dataset settings
train_list = [
    _base_.ic11_rec_train, _base_.ic13_rec_train, _base_.ic15_rec_train,
    _base_.cocov1_rec_train, _base_.iiit5k_rec_train, _base_.mj_sub_rec_train,
    _base_.st_sub_rec_train, _base_.st_add_rec_train
]
test_list = [
    _base_.cute80_rec_test, _base_.iiit5k_rec_test, _base_.svt_rec_test,
    _base_.svtp_rec_test, _base_.ic13_rec_test, _base_.ic15_rec_test
]

train_list = [
    dict(
        type='RepeatDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=train_list[:5],
            pipeline=train_pipeline),
        times=20),
    dict(
        type='ConcatDataset', datasets=train_list[5:],
        pipeline=train_pipeline),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='ConcatDataset', datasets=train_list))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=test_list, pipeline=test_pipeline))
val_dataloader = test_dataloader

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric')
    ],
    dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
test_evaluator = val_evaluator

visualizer = dict(type='TextRecogLocalVisualizer', name='visualizer')
