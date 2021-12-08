_base_ = ['../../_base_/default_runtime.py']

num_chars = 37
max_seq_len = 26
label_convertor = dict(
    type='ABIConvertor',
    dict_type='DICT36',
    with_unknown=False,
    with_padding=False,
    lower=True,
)

model = dict(
    type='ABINet',
    backbone=dict(type='ResNetABI'),
    encoder=dict(
        type='ABIVisionModel',
        encoder=dict(
            type='ResTransformer',
            n_layers=3,
            n_head=8,
            d_model=512,
            d_inner=2048,
            dropout=0.1,
            max_len=8 * 32,
        ),
        decoder=dict(
            type='ABIVisionDecoder',
            in_channels=512,
            num_channels=64,
            attn_height=8,
            attn_width=32,
            attn_mode='nearest',
            use_result='feature',
            num_chars=num_chars,
            max_seq_len=max_seq_len,
            init_cfg=dict(type='Xavier', layer='Conv2d')),
    ),
    decoder=dict(
        type='ABILanguageDecoder',
        d_model=512,
        n_head=8,
        d_inner=2048,
        n_layers=4,
        dropout=0.1,
        detach_tokens=True,
        use_self_attn=False,
        pad_idx=num_chars - 1,
        num_chars=num_chars,
        max_seq_len=max_seq_len,
        init_cfg=None),
    fuser=dict(
        type='BaseAlignment',
        d_model=512,
        num_chars=num_chars,
        init_cfg=None,
        max_seq_len=max_seq_len,
    ),
    loss=dict(
        type='ABILoss', enc_weight=1.0, dec_weight=1.0, fusion_weight=1.0),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len,
    iter_size=3)

# optimizer
optimizer = dict(type='Adam', lr=4e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[5, 6])
total_epochs = 6

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=128,
        max_width=128,
        keep_aspect_ratio=False,
        width_downsample_ratio=0.25),
    dict(
        type='RandomWrapper',
        p=0.5,
        transforms=[
            dict(
                type='OneOfWrapper',
                transforms=[
                    dict(
                        type='RandomRotateTextDet',
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
        type='RandomWrapper',
        p=0.25,
        transforms=[
            dict(type='PyramidRescale'),
            dict(
                type='Albu',
                transforms=[
                    dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
                    dict(type='MotionBlur', blur_limit=6, p=0.5),
                ]),
        ]),
    dict(
        type='RandomWrapper',
        p=0.25,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1),
        ]),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio',
            'resize_shape'
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
                min_width=128,
                max_width=128,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'resize_shape'
                ]),
        ])
]

dataset_type = 'OCRDataset'

train_prefix = 'data/mixture/'

train_img_prefix1 = train_prefix + \
    'SynthText/synthtext/SynthText_patch_horizontal'
train_img_prefix2 = train_prefix + 'Syn90k/mnt/ramdisk/max/90kDICT32px'

train_ann_file1 = train_prefix + 'SynthText/label.lmdb'
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
    pipeline=None,
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
    samples_per_gpu=256,
    workers_per_gpu=6,
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
