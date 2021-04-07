_base_ = [
    '../../_base_/default_runtime.py', '../../_base_/recog_models/nrtr.py'
]

label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)

model = dict(
    type='NRTR',
    backbone=dict(
        type='ResNet31OCR',
        layers=[1, 2, 5, 3],
        channels=[32, 64, 128, 256, 512, 512],
        stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
        last_stage_pool=False),
    encoder=dict(type='TFEncoder'),
    decoder=dict(type='TFDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=40)

# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 6

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=160,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=32,
                min_width=32,
                max_width=160,
                keep_aspect_ratio=True,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio'
                ]),
        ])
]

dataset_type = 'OCRDataset'

train_prefix = 'data/mixture/'

train_img_prefix1 = train_prefix + \
    'SynthText/synthtext/SynthText_patch_horizontal'
train_img_prefix2 = train_prefix + 'Syn90k/mnt/ramdisk/max/90kDICT32px'

train_ann_file1 = train_prefix + 'SynthText/label.lmdb',
train_ann_file2 = train_prefix + 'Syn90k/label.lmdb'

train1 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix1,
    ann_file=train_ann_file1,
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

train2 = {key: value for key, value in train1.items()}
train2['img_prefix'] = train_img_prefix2
train2['ann_file'] = train_ann_file2

test_prefix = 'data/mixture/'
test_img_prefix1 = test_prefix + 'IIIT5K/'
test_img_prefix2 = test_prefix + 'svt/'
test_img_prefix3 = test_prefix + 'icdar_2013/'
test_img_prefix4 = test_prefix + 'icdar_2015/'
test_img_prefix5 = test_prefix + 'svtp/'
test_img_prefix6 = test_prefix + 'ct80/'

test_ann_file1 = test_prefix + 'IIIT5K/test_label.txt'
test_ann_file2 = test_prefix + 'svt/test_label.txt'
test_ann_file3 = test_prefix + 'icdar_2013/test_label_1015.txt'
test_ann_file4 = test_prefix + 'icdar_2015/test_label.txt'
test_ann_file5 = test_prefix + 'svtp/test_label.txt'
test_ann_file6 = test_prefix + 'ct80/test_label.txt'

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

test5 = {key: value for key, value in test1.items()}
test5['img_prefix'] = test_img_prefix5
test5['ann_file'] = test_ann_file5

test6 = {key: value for key, value in test1.items()}
test6['img_prefix'] = test_img_prefix6
test6['ann_file'] = test_ann_file6

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(type='ConcatDataset', datasets=[train1, train2]),
    val=dict(
        type='ConcatDataset',
        datasets=[test1, test2, test3, test4, test5, test6]),
    test=dict(
        type='ConcatDataset',
        datasets=[test1, test2, test3, test4, test5, test6]))

evaluation = dict(interval=1, metric='acc')
