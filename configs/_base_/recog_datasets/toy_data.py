data_root = 'tests/data/rec_toy_dataset'
train_img_prefix = 'imgs/'
train_anno_file = 'label.json'

train_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix),
    ann_file=train_anno_file,
    pipeline=None,
    test_mode=False)

test_anno_file = f'{data_root}/label.json'
test_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix),
    ann_file=train_anno_file,
    pipeline=None,
    test_mode=True)

train_list = [train_dataset]

test_list = [test_dataset]
