_base_ = [
    '../../_base_/default_runtime.py', '../../_base_/recog_models/nrtr.py'
]

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
            'filename', 'ori_shape', 'resize_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
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
        meta_keys=['filename', 'ori_shape', 'resize_shape', 'valid_ratio'])
]

dataset_type = 'OCRDataset'

data_root = 'data/mixture'

train_img_prefix1 = f'{data_root}/Syn90k/mnt/ramdisk/max/90kDICT32px'
train_ann_file1 = f'{data_root}/Syn90k/label.lmdb'

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
    pipeline=None,
    test_mode=False)

train_img_prefix2 = f'{data_root}/SynthText/' + \
    'synthtext/SynthText_patch_horizontal'
train_ann_file2 = f'{data_root}/SynthText/label.lmdb'

train2 = {key: value for key, value in train1.items()}
train2['img_prefix'] = train_img_prefix2
train2['ann_file'] = train_ann_file2

test_img_prefix1 = f'{data_root}/IIIT5K/'
test_img_prefix2 = f'{data_root}/svt/'
test_img_prefix3 = f'{data_root}/icdar_2013/'
test_img_prefix4 = f'{data_root}/icdar_2015/'
test_img_prefix5 = f'{data_root}/svtp/'
test_img_prefix6 = f'{data_root}/ct80/'

test_ann_file1 = f'{data_root}/IIIT5K/test_label.txt'
test_ann_file2 = f'{data_root}/svt/test_label.txt'
test_ann_file3 = f'{data_root}/icdar_2013/test_label_1015.txt'
test_ann_file4 = f'{data_root}/icdar_2015/test_label.txt'
test_ann_file5 = f'{data_root}/svtp/test_label.txt'
test_ann_file6 = f'{data_root}/ct80/test_label.txt'

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
    pipeline=None,
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
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1, train2],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=[test1, test2, test3, test4, test5, test6],
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=[test1, test2, test3, test4, test5, test6],
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
