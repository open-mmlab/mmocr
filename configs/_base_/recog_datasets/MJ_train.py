# Text Recognition Training set, including:
# Synthetic Datasets: Syn90k

mj_rec_train = dict(
    type='OCRDataset',
    data_root='data/rec',
    data_prefix=dict(img_path='Syn90k/mnt/ramdisk/max/90kDICT32px'),
    ann_file='Syn90k/train_labels.json',
    test_mode=False,
    pipeline=None)

train_list = [mj_rec_train]
