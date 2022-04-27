dataset_type = 'TextDetDataset'
data_root = 'data/synthtext'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.lmdb',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

train_list = [train]
test_list = [train]
