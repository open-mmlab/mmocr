_base_ = [
    '_base_mask-rcnn_resnet50_fpn.py',
    '../../_base_/det_datasets/ctw1500.py',
    '../../_base_/textdet_default_runtime.py',
    '../../_base_/schedules/schedule_sgd_160e.py',
]

# dataset settings
ctw_det_train = _base_.ctw_det_train
ctw_det_test = _base_.ctw_det_test

# test pipeline for CTW1500
ctw_test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1600, 1600), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

ctw_det_train.pipeline = _base_.train_pipeline
ctw_det_test.pipeline = ctw_test_pipeline

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ctw_det_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ctw_det_test)

test_dataloader = val_dataloader
