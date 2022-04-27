dataset_type = 'OCRDataset'

root = 'tests/data/ocr_toy_dataset'
img_prefix = f'{root}/imgs'
train_anno_file1 = f'{root}/label.txt'

train1 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file1,
    loader=dict(
        type='AnnFileLoader',
        repeat=100,
        file_format='txt',
        file_storage_backend='disk',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train_anno_file2 = f'{root}/label.lmdb'
train2 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file2,
    loader=dict(
        type='AnnFileLoader',
        repeat=100,
        file_format='lmdb',
        file_storage_backend='disk',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

test_anno_file1 = f'{root}/label.lmdb'
test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=test_anno_file1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        file_storage_backend='disk',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

train_list = [train1, train2]

test_list = [test]
