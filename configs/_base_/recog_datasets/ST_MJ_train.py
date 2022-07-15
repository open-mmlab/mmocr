# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, Syn90k

data_root = 'data/recog'

train_img_prefix1 = 'Syn90k/mnt/ramdisk/max/90kDICT32px'
train_ann_file1 = 'Syn90k/label.json'
file_client_args = dict(backend='disk')

train1 = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix1),
    ann_file=train_ann_file1,
    test_mode=False,
    pipeline=None)

train_img_prefix2 = 'SynthText/synthtext/SynthText_patch_horizontal'
train_ann_file2 = 'SynthText/label.json'
train2 = {key: value for key, value in train1.items()}
train2['data_root'] = data_root
train2['data_prefix'] = dict(img_path=train_img_prefix2),
train2['ann_file'] = dict(img_path=train_ann_file2),

train_list = [train1, train2]
