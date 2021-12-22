# Copyright (c) OpenMMLab. All rights reserved.
# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, SynthAdd, Syn90k
# Real Dataset: IC11, IC13, IC15, COCO-Test, IIIT5k

train_prefix = 'data/mixture'

train_img_prefix1 = f'{train_prefix}/icdar_2011'
train_img_prefix2 = f'{train_prefix}/icdar_2013'
train_img_prefix3 = f'{train_prefix}/icdar_2015'
train_img_prefix4 = f'{train_prefix}/coco_text'
train_img_prefix5 = f'{train_prefix}/IIIT5K'
train_img_prefix6 = f'{train_prefix}/SynthText_Add'
train_img_prefix7 = f'{train_prefix}/SynthText'
train_img_prefix8 = f'{train_prefix}/Syn90k'

train_ann_file1 = f'{train_prefix}/icdar_2011/train_label.txt',
train_ann_file2 = f'{train_prefix}/icdar_2013/train_label.txt',
train_ann_file3 = f'{train_prefix}/icdar_2015/train_label.txt',
train_ann_file4 = f'{train_prefix}/coco_text/train_label.txt',
train_ann_file5 = f'{train_prefix}/IIIT5K/train_label.txt',
train_ann_file6 = f'{train_prefix}/SynthText_Add/label.txt',
train_ann_file7 = f'{train_prefix}/SynthText/shuffle_labels.txt',
train_ann_file8 = f'{train_prefix}/Syn90k/shuffle_labels.txt'

train1 = dict(
    type='OCRDataset',
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
    pipeline=None,
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
    type='OCRDataset',
    img_prefix=train_img_prefix6,
    ann_file=train_ann_file6,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train7 = {key: value for key, value in train6.items()}
train7['img_prefix'] = train_img_prefix7
train7['ann_file'] = train_ann_file7

train8 = {key: value for key, value in train6.items()}
train8['img_prefix'] = train_img_prefix8
train8['ann_file'] = train_ann_file8

train_list = [train1, train2, train3, train4, train5, train6, train7, train8]
