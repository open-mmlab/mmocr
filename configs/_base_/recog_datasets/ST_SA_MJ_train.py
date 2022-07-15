# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, Syn90k
data_root = 'data/recog'

train_img_prefix1 = 'SynthText_Add'
train_img_prefix2 = 'SynthText/synthtext/' + \
    'SynthText_patch_horizontal'
train_img_prefix3 = 'Syn90k/mnt/ramdisk/max/90kDICT32px'

train_ann_file1 = 'SynthText_Add/label.json',
train_ann_file2 = 'SynthText/label.json',
train_ann_file3 = 'Syn90k/label.json'

train1 = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix1),
    ann_file=train_ann_file1,
    test_mode=False,
    pipeline=None)

train2 = {key: value for key, value in train1.items()}
train2['data_prefix'] = dict(img_path=train_img_prefix2)
train2['ann_file'] = train_ann_file2

train3 = {key: value for key, value in train1.items()}
train3['img_prefix'] = dict(img_path=train_img_prefix3)
train3['ann_file'] = train_ann_file3

train_list = [train1, train2, train3]
