data_root = 'data/funsd'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://guillaumejaume.github.io/FUNSD/dataset.zip',
            save_name='funsd.zip',
            md5='e05de47de238aa343bf55d8807d659a9',
            split=['train', 'test'],
            content=['image', 'annotation'],
            mapping=[
                ['funsd/dataset/training_data/images', 'textdet_imgs/train'],
                ['funsd/dataset/testing_data/images', 'textdet_imgs/test'],
                [
                    'funsd/dataset/training_data/annotations',
                    'annotations/train'
                ],
                ['funsd/dataset/testing_data/annotations', 'annotations/test'],
            ]),
    ])

data_converter = dict(
    type='TextDetDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='pair_gather',
        suffixes=['.png'],
        rule=[r'(\w+)\.png', r'\1.json']),
    parser=dict(type='FUNSDTextDetAnnParser'),
    dumper=dict(type='JsonDumper'),
    delete=['annotations', 'funsd'])
