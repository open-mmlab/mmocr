_base_ = [
    '_base_psenet_resnet50_fpnf.py',
    '../_base_/datasets/ctw1500.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
]

# optimizer
optim_wrapper = dict(optimizer=dict(lr=1e-4))
train_cfg = dict(val_interval=40)
param_scheduler = [
    dict(type='MultiStepLR', milestones=[200, 400], end=600),
]

# dataset settings
ctw1500_textdet_train = _base_.ctw1500_textdet_train
ctw1500_textdet_test = _base_.ctw1500_textdet_test

test_pipeline_ctw = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# pipeline settings
ctw1500_textdet_train.pipeline = _base_.train_pipeline
ctw1500_textdet_test.pipeline = test_pipeline_ctw

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ctw1500_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ctw1500_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=64 * 4)
