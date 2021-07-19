_base_ = []
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook')

    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# model
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT36', with_unknown=False, lower=True)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)

train_cfg = None
test_cfg = None

# optimizer
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
total_epochs = 5

# data
img_norm_cfg = dict(mean=[0.5], std=[0.5])

train_pipeline = [
    dict(type='LoadOCRImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False),
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
    dict(type='LoadOCRImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=4,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'ori_shape', 'img_shape', 'valid_ratio']),
]

dataset_type = 'OCRDataset'

train_img_prefix = 'data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px'
train_ann_file = 'data/mixture/Syn90k/label.lmdb'

train1 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix,
    ann_file=train_ann_file,
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
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1],
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

cudnn_benchmark = True
