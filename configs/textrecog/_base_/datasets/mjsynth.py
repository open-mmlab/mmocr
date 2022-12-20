mjsynth_textrecog_data_root = 'data/rec/Syn90k/'

mjsynth_textrecog_test = dict(
    type='OCRDataset',
    data_root=mjsynth_textrecog_data_root,
    data_prefix=dict(img_path='mnt/ramdisk/max/90kDICT32px'),
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

mjsynth_sub_textrecog_train = dict(
    type='OCRDataset',
    data_root=mjsynth_textrecog_data_root,
    data_prefix=dict(img_path='mnt/ramdisk/max/90kDICT32px'),
    ann_file='subset_train_labels.json',
    test_mode=False,
    pipeline=None)
