_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_20e.py',
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=100))

# dataset settings
dataset_type = 'OCRDataset'
data_root = 'data/recog/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(128, 32)),
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(
                        type='RandomRotate',
                        max_angle=15,
                    ),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAffine',
                        degrees=15,
                        translate=(0.3, 0.3),
                        scale=(0.5, 2.),
                        shear=(-45, 45),
                    ),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomPerspective',
                        distortion_scale=0.5,
                        p=1,
                    ),
                ])
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(type='PyramidRescale'),
            dict(
                type='mmdet.Albu',
                transforms=[
                    dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
                    dict(type='MotionBlur', blur_limit=6, p=0.5),
                ]),
        ]),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1),
        ]),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(128, 32)),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio',
                   'instances'))
]

dataset_mj = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='mnt/ramdisk/max/90kDICT32px/'),
    ann_file='data/MJ/label.json',
    pipeline=train_pipeline)
dataset_st = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='SynthText/synthtext/SynthText_patch_horizontal/'),
    ann_file='data/ST/alphanumeric_labels.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=192 * 4,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='ConcatDataset', datasets=[dataset_mj, dataset_st]))

val_dataloader = dict(
    batch_size=192,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='testset/testset/IIIT5K/'),
        ann_file='label.json',
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='WordMetric', mode=['ignore_case_symbol'])
test_evaluator = val_evaluator
visualizer = dict(type='TextRecogLocalVisualizer', name='visualizer')
