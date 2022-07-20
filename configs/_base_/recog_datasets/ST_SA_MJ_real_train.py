# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, SynthAdd, Syn90k
# Real Dataset: IC11, IC13, IC15, COCO-Test, IIIT5k
data_root = 'data/rec'

train_img_prefix1 = 'icdar_2011'
train_img_prefix2 = 'icdar_2013'
train_img_prefix3 = 'icdar_2015'
train_img_prefix4 = 'coco_text'
train_img_prefix5 = 'IIIT5K'
train_img_prefix6 = 'SynthText_Add'
train_img_prefix7 = 'SynthText/synthtext/SynthText_patch_horizontal'
train_img_prefix8 = 'Syn90k/mnt/ramdisk/max/90kDICT32px'

train_ann_file1 = 'icdar_2011/train_label.json',
train_ann_file2 = 'icdar_2013/train_label.json',
train_ann_file3 = 'icdar_2015/train_label.json',
train_ann_file4 = 'coco_text/train_label.json',
train_ann_file5 = 'IIIT5K/train_label.json',
train_ann_file6 = 'SynthText_Add/train_label.json',
train_ann_file7 = 'SynthText/shuffle_label.json',
train_ann_file8 = 'Syn90k/mnt/ramdisk/max/90kDICT32px/shuffle_label.json'

IC11 = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix1),
    ann_file=train_ann_file1,
    test_mode=False,
    pipeline=None)

IC13 = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix2),
    ann_file=train_ann_file2,
    test_mode=False,
    pipeline=None)

IC15 = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix3),
    ann_file=train_ann_file3,
    test_mode=False,
    pipeline=None)

COCO = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix4),
    ann_file=train_ann_file4,
    test_mode=False,
    pipeline=None)

IIIT5K = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix5),
    ann_file=train_ann_file5,
    test_mode=False,
    pipeline=None)

STADD = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix6),
    ann_file=train_ann_file6,
    test_mode=False,
    pipeline=None)

ST = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix7),
    ann_file=train_ann_file7,
    test_mode=False,
    pipeline=None)

MJ = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix8),
    ann_file=train_ann_file8,
    test_mode=False,
    pipeline=None)

train_list = [IC13, IC11, IC15, COCO, IIIT5K, STADD, ST, MJ]
