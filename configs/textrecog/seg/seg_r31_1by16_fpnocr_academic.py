_base_ = ['../../_base_/default_runtime.py']

# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 5

label_convertor = dict(
    type='SegConvertor', dict_type='DICT36', with_unknown=True, lower=True)

model = dict(
    type='SegRecognizer',
    backbone=dict(
        type='ResNet31OCR',
        layers=[1, 2, 5, 3],
        channels=[32, 64, 128, 256, 512, 512],
        out_indices=[0, 1, 2, 3],
        stage4_pool_cfg=dict(kernel_size=2, stride=2),
        last_stage_pool=True),
    neck=dict(
        type='FPNOCR', in_channels=[128, 256, 512, 512], out_channels=256),
    head=dict(
        type='SegHead',
        in_channels=256,
        upsample_param=dict(scale_factor=2.0, mode='nearest')),
    loss=dict(
        type='SegLoss', seg_downsample_ratio=1.0, seg_with_loss_weight=True),
    label_convertor=label_convertor)

find_unused_parameters = True

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

gt_label_convertor = dict(
    type='SegConvertor', dict_type='DICT36', with_unknown=True, lower=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomPaddingOCR',
        max_ratio=[0.15, 0.2, 0.15, 0.2],
        box_type='char_quads'),
    dict(type='OpencvToPil'),
    dict(
        type='RandomRotateImageBox',
        min_angle=-17,
        max_angle=17,
        box_type='char_quads'),
    dict(type='PilToOpencv'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=512,
        keep_aspect_ratio=True),
    dict(
        type='OCRSegTargets',
        label_convertor=gt_label_convertor,
        box_type='char_quads'),
    dict(type='RandomRotateTextDet', rotate_ratio=0.5, max_angle=15),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='ToTensorOCR'),
    dict(type='FancyPCA'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels'],
        visualize=dict(flag=False, boundary_key=None),
        call_super=False),
    dict(
        type='Collect',
        keys=['img', 'gt_kernels'],
        meta_keys=['filename', 'ori_shape', 'img_shape'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(type='CustomFormatBundle', call_super=False),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'ori_shape', 'img_shape'])
]

train_img_root = 'data/mixture/'

train_img_prefix = train_img_root + 'SynthText'

train_ann_file = train_img_root + 'SynthText/instances_train.txt'

train = dict(
    type='OCRSegDataset',
    img_prefix=train_img_prefix,
    ann_file=train_ann_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser', keys=['file_name', 'annotations', 'text'])),
    pipeline=train_pipeline,
    test_mode=False)

dataset_type = 'OCRDataset'
test_prefix = 'data/mixture/'

test_img_prefix1 = test_prefix + 'IIIT5K/'
test_img_prefix2 = test_prefix + 'svt/'
test_img_prefix3 = test_prefix + 'icdar_2013/'
test_img_prefix4 = test_prefix + 'ct80/'

test_ann_file1 = test_prefix + 'IIIT5K/test_label.txt'
test_ann_file2 = test_prefix + 'svt/test_label.txt'
test_ann_file3 = test_prefix + 'icdar_2013/test_label_1015.txt'
test_ann_file4 = test_prefix + 'ct80/test_label.txt'

test1 = dict(
    type=dataset_type,
    img_prefix=test_img_prefix1,
    ann_file=test_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=test_pipeline,
    test_mode=True)

test2 = {key: value for key, value in test1.items()}
test2['img_prefix'] = test_img_prefix2
test2['ann_file'] = test_ann_file2

test3 = {key: value for key, value in test1.items()}
test3['img_prefix'] = test_img_prefix3
test3['ann_file'] = test_ann_file3

test4 = {key: value for key, value in test1.items()}
test4['img_prefix'] = test_img_prefix4
test4['ann_file'] = test_ann_file4

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train]),
    val=dict(type='ConcatDataset', datasets=[test1, test2, test3, test4]),
    test=dict(type='ConcatDataset', datasets=[test1, test2, test3, test4]))

evaluation = dict(interval=1, metric='acc')
