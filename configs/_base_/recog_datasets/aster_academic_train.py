train_img_root = '/data/mixture/'

train_img_prefix1 = f'{train_img_root}/Syn90k'
train_img_prefix2 = f'{train_img_root}/SynthText'
train_ann_file1 = f'{train_img_root}/Syn90k/shuffle_labels.txt'
train_ann_file2 = f'{train_img_root}/SynthText/shuffle_labels.txt'

train1 = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix1,
    ann_file=train_ann_file1,
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

train2 = {key: value for key, value in train1.items()}
train_list = [train1, train2]
