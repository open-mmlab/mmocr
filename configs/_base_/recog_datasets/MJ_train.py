# Text Recognition Training set, including:
# Synthetic Datasets: Syn90k

train_root = 'data/mixture/Syn90k'

train_img_prefix = f'{train_root}/mnt/ramdisk/max/90kDICT32px'
train_ann_file = f'{train_root}/label.lmdb'

train = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix,
    ann_file=train_ann_file,
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

train_list = [train]
