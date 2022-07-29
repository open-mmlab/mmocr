# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, SynthAdd, Syn90k
# Real Dataset: IC11, IC13, IC15, COCO-Test, IIIT5k
data_root = 'data/rec'

train_img_prefix1 = 'icdar_2011'
train_img_prefix2 = 'icdar_2013'
train_img_prefix3 = 'icdar_2015'
train_img_prefix4 = 'coco_text_v1'
train_img_prefix5 = 'IIIT5K'
train_img_prefix6 = 'synthtext_add'
train_img_prefix7 = 'SynthText/synthtext/SynthText_patch_horizontal'
train_img_prefix8 = 'Syn90k/mnt/ramdisk/max/90kDICT32px'

train_ann_file1 = 'icdar_2011/train_labels.json'
train_ann_file2 = 'icdar_2013/train_labels.json'
train_ann_file3 = 'icdar_2015/train_labels.json'
train_ann_file4 = 'coco_text_v1/train_labels.json'
train_ann_file5 = 'IIIT5K/train_labels.json'
train_ann_file6 = 'synthtext_add/train_labels.json'
train_ann_file7 = 'SynthText/shuffle_train_labels.json'
train_ann_file8 = 'Syn90k/shuffle_train_labels.json'

ic11_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix1),
    ann_file=train_ann_file1,
    test_mode=False,
    pipeline=None)

ic13_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix2),
    ann_file=train_ann_file2,
    test_mode=False,
    pipeline=None)

ic15_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix3),
    ann_file=train_ann_file3,
    test_mode=False,
    pipeline=None)

cocov1_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix4),
    ann_file=train_ann_file4,
    test_mode=False,
    pipeline=None)

iiit5k_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix5),
    ann_file=train_ann_file5,
    test_mode=False,
    pipeline=None)

st_add_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix6),
    ann_file=train_ann_file6,
    test_mode=False,
    pipeline=None)

st_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix7),
    ann_file=train_ann_file7,
    test_mode=False,
    pipeline=None)

mj_rec_trian = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix8),
    ann_file=train_ann_file8,
    test_mode=False,
    pipeline=None)

train_list = [
    ic13_rec_train, ic11_rec_train, ic15_rec_train, cocov1_rec_train,
    iiit5k_rec_train, st_add_rec_train, st_rec_train, mj_rec_trian
]
