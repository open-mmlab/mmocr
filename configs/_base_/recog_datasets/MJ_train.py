# Text Recognition Training set, including:
# Synthetic Datasets: Syn90k
data_root = 'data/recog'
train_img_prefix1 = 'Syn90k/mnt/ramdisk/max/90kDICT32px'
train_ann_file1 = 'Syn90k/label.json'
file_client_args = dict(backend='disk')

train = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix1),
    ann_file=train_ann_file1,
    test_mode=False,
    pipeline=None)
train_list = [train]
