# data
img_norm_cfg = dict(mean=[0.5], std=[0.5])

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=256,
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
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
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

train_prefix = 'data/chineseocr/'

train_ann_file1 = train_prefix + 'labels/label_digits_train.txt',
train_ann_file2 = train_prefix + 'labels/label_handwriting_train.txt',
train_ann_file3 = train_prefix +\
    'labels/label_printed_chinese_english_digits_train.txt',
train_ann_file4 = train_prefix + 'labels/label_signatures_train.txt',

train1 = dict(
    type=dataset_type,
    img_prefix=train_prefix,
    ann_file=train_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

train2 = {key: value for key, value in train1.items()}
train2['ann_file'] = train_ann_file2

train3 = {key: value for key, value in train1.items()}
train3['ann_file'] = train_ann_file3

train4 = {key: value for key, value in train1.items()}
train4['ann_file'] = train_ann_file4

test_prefix = 'data/chineseocr/'

test_ann_file1 = train_prefix + 'labels/label_digits_test.txt',
test_ann_file2 = train_prefix + 'labels/label_handwriting_test.txt',
test_ann_file3 = train_prefix +\
    'labels/label_printed_chinese_english_digits_test.txt',
test_ann_file4 = train_prefix + 'labels/label_signatures_test.txt',

test1 = dict(
    type=dataset_type,
    img_prefix=test_prefix,
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
test2['ann_file'] = test_ann_file2

test3 = {key: value for key, value in test1.items()}
test3['ann_file'] = test_ann_file3

test4 = {key: value for key, value in test1.items()}
test4['ann_file'] = test_ann_file4

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train3]),
    val=dict(type='ConcatDataset', datasets=[test3]),
    test=dict(type='ConcatDataset', datasets=[test3]))

evaluation = dict(interval=1, metric='acc')

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
    type='CTCConvertor',
    dict_file='data/chineseocr/labels/dict_printed_chinese_english_digits.txt',
    with_unknown=True,
    lower=False)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss', zero_infinity=True, flatten=False),
    label_convertor=label_convertor,
    pretrained=None)

train_cfg = None
test_cfg = None

# optimizer
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
total_epochs = 50

cudnn_benchmark = True
