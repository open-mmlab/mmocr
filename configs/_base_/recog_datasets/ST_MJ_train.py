# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, Syn90k

train_root = 'data/mixture'

train_img_prefix1 = f'{train_root}/Syn90k/mnt/ramdisk/max/90kDICT32px'
train_ann_file1 = f'{train_root}/Syn90k/label.lmdb'

train1 = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix1,
    ann_file=train_ann_file1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train_img_prefix2 = f'{train_root}/SynthText/' + \
    'synthtext/SynthText_patch_horizontal'
train_ann_file2 = f'{train_root}/SynthText/label.lmdb'

train2 = {key: value for key, value in train1.items()}
train2['img_prefix'] = train_img_prefix2
train2['ann_file'] = train_ann_file2

train_list = [train1, train2]
