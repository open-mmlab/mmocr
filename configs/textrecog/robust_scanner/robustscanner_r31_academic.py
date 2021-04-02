_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_models/robust_scanner.py'
]

# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 5

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
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
                height=48,
                min_width=48,
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

prefix = 'data/mixture/'

train_img_prefix1 = prefix + 'icdar_2011/Challenge1_Training_Task3_Images_GT'
train_img_prefix2 = prefix + 'icdar_2013/recog_train_data/' + \
    'Challenge2_Training_Task3_Images_GT'
train_img_prefix3 = prefix + 'icdar_2015/ch4_training_word_images_gt'
train_img_prefix4 = prefix + 'coco_text/train_words'
train_img_prefix5 = prefix + 'III5K'
train_img_prefix6 = prefix + 'SynthText_Add/SynthText_Add'
train_img_prefix7 = prefix + 'SynthText/synthtext/SynthText_patch_horizontal'
train_img_prefix8 = prefix + 'mnt/ramdisk/max/90kDICT32px'

train_ann_file1 = prefix + 'icdar_2011/training_label_fix.txt',
train_ann_file2 = prefix + 'icdar_2013/recog_train_data/train_label.txt',
train_ann_file3 = prefix + 'icdar_2015/training_label_fix.txt',
train_ann_file4 = prefix + 'coco_text/train_label.txt',
train_ann_file5 = prefix + 'III5K/train_label.txt',
train_ann_file6 = prefix + 'SynthText_Add/SynthText_Add/' + \
    'annotationlist/label.lmdb',
train_ann_file7 = prefix + 'SynthText/synthtext/shuffle_labels.lmdb',
train_ann_file8 = prefix + 'mnt/ramdisk/max/90kDICT32px/shuffle_labels.lmdb'

train1 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix1,
    ann_file=train_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=20,
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

train3 = {key: value for key, value in train1.items()}
train3['img_prefix'] = train_img_prefix3
train3['ann_file'] = train_ann_file3

train4 = {key: value for key, value in train1.items()}
train4['img_prefix'] = train_img_prefix4
train4['ann_file'] = train_ann_file4

train5 = {key: value for key, value in train1.items()}
train5['img_prefix'] = train_img_prefix5
train5['ann_file'] = train_ann_file5

train6 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix6,
    ann_file=train_ann_file6,
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

train7 = {key: value for key, value in train6.items()}
train7['img_prefix'] = train_img_prefix7
train7['ann_file'] = train_ann_file7

train8 = {key: value for key, value in train6.items()}
train8['img_prefix'] = train_img_prefix8
train8['ann_file'] = train_ann_file8

test_img_prefix1 = prefix + 'testset/IIIT5K/'
test_img_prefix2 = prefix + 'testset/svt/'
test_img_prefix3 = prefix + 'testset/icdar_2013/Challenge2_Test_Task3_Images/'
test_img_prefix4 = prefix + 'testset/icdar_2015/ch4_test_word_images_gt'
test_img_prefix5 = prefix + 'testset/svtp/'
test_img_prefix6 = prefix + 'testset/ct80/'

test_ann_file1 = prefix + 'testset/IIIT5K/label.txt'
test_ann_file2 = prefix + 'testset/svt/test_list.txt'
test_ann_file3 = prefix + 'testset/icdar_2013/test_label_1015.txt'
test_ann_file4 = prefix + 'testset/icdar_2015/test_label.txt'
test_ann_file5 = prefix + 'testset/svtp/imagelist.txt'
test_ann_file6 = prefix + 'testset/ct80/imagelist.txt'

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
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[
            train1, train2, train3, train4, train5, train6, train7, train8
        ]),
    val=dict(
        type='ConcatDataset',
        datasets=[test1, test2, test3, test4, test5, test6]),
    test=dict(
        type='ConcatDataset',
        datasets=[test1, test2, test3, test4, test5, test6]))

evaluation = dict(interval=1, metric='acc')
