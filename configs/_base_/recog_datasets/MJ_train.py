# Text Recognition Training set, including:
# Synthetic Datasets: Syn90k
data_root = 'data/rec'
train_img_prefix1 = 'Syn90k/mnt/ramdisk/max/90kDICT32px'
train_ann_file1 = 'Syn90k/label.json'

mj_rec_train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix1),
    ann_file=train_ann_file1,
    test_mode=False,
    pipeline=None)
train_list = [mj_rec_train]
