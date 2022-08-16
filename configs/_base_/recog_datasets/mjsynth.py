mj_rec_data_root = 'data/rec/'

mj_rec_train = dict(
    type='OCRDataset',
    data_root=mj_rec_data_root,
    data_prefix=dict(img_path='mnt/ramdisk/max/90kDICT32px'),
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

mj_sub_rec_train = dict(
    type='OCRDataset',
    data_root=mj_rec_data_root,
    data_prefix=dict(img_path='mnt/ramdisk/max/90kDICT32px'),
    ann_file='subset_train_labels.json',
    test_mode=False,
    pipeline=None)
