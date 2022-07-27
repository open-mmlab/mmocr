# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, Syn90k
data_root = 'data/rec'

mj_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path='Syn90k/mnt/ramdisk/max/90kDICT32px'),
    ann_file='Syn90k/train_labels.json',
    test_mode=False,
    pipeline=None)

st_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(
        img_path='SynthText/synthtext/SynthText_patch_horizontal'),
    ann_file='SynthText/train_labels.json',
    test_mode=False,
    pipeline=None)

st_add_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path='synthtext_add'),
    ann_file='synthtext_add/train_labels.json',
    test_mode=False,
    pipeline=None)

train_list = [mj_rec_train, st_rec_train, st_add_rec_train]
