_base_ = [
    '_base_svtr-tiny.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/mjsynth.py',
    '../_base_/datasets/synthtext.py',
    '../_base_/datasets/cute80.py',
    '../_base_/datasets/iiit5k.py',
    '../_base_/datasets/svt.py',
    '../_base_/datasets/svtp.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/schedules/schedule_adam_base.py',
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5 / (10**4) * 2048 / 2048,
        betas=(0.9, 0.99),
        eps=8e-8,
        weight_decay=0.05))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.5,
        end_factor=1.,
        end=2,
        verbose=False,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=19,
        begin=2,
        end=20,
        verbose=False,
        convert_to_iter_based=True),
]

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ignore_empty=True,
        min_size=5),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='TextRecogGeneralAug', ),
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='TextRecogRandomCrop', ),
        ],
    ),
    dict(
        type='ConditionApply',
        condition='min(results["img_shape"])>10',
        transforms=dict(
            type='RandomApply',
            prob=0.4,
            transforms=[
                dict(
                    type='TorchVisionWrapper',
                    op='GaussianBlur',
                    kernel_size=5,
                    sigma=1,
                ),
            ],
        )),
    dict(
        type='RandomApply',
        prob=0.4,
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
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='ImageContentJitter', ),
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(
                type='ImgAugWrapper',
                args=[dict(cls='AdditiveGaussianNoise', scale=0.1**0.5)]),
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='ReversePixels', ),
        ],
    ),
    dict(type='Resize', scale=(256, 64)),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(256, 64)),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

# dataset settings
train_list = [_base_.mj_rec_train, _base_.st_rec_train]
test_list = [
    _base_.cute80_rec_test, _base_.iiit5k_rec_test, _base_.svt_rec_test,
    _base_.svtp_rec_test, _base_.ic13_rec_test, _base_.ic15_rec_test
]

train_dataloader = dict(
    batch_size=512,
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset', datasets=train_list, pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=test_list, pipeline=test_pipeline))

test_dataloader = val_dataloader
