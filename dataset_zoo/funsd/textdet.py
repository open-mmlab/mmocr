data_root = 'data/funsd'
cache_path = 'data/cache'

prepare_train_data = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        data_root=data_root,
        files=[
            dict(
                url='https://guillaumejaume.github.io/FUNSD/dataset.zip',
                save_name='funsd.zip',
                md5='e05de47de238aa343bf55d8807d659a9',
                content=['image', 'annotation'],
                mapping=[
                    [
                        'funsd/dataset/training_data/images',
                        'textdet_imgs/train'
                    ],
                    [
                        'funsd/dataset/training_data/annotations',
                        'annotations/train'
                    ],
                ]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.png'],
        rule=[r'(\w+)\.png', r'\1.json']),
    parser=dict(type='FUNSDTextDetAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

prepare_test_data = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://guillaumejaume.github.io/FUNSD/dataset.zip',
                save_name='funsd.zip',
                md5='e05de47de238aa343bf55d8807d659a9',
                content=['image', 'annotation'],
                mapping=[
                    ['funsd/dataset/testing_data/images', 'textdet_imgs/test'],
                    [
                        'funsd/dataset/testing_data/annotations',
                        'annotations/test'
                    ],
                ]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.png'],
        rule=[r'(\w+)\.png', r'\1.json']),
    parser=dict(type='FUNSDTextDetAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

config_generator = dict(type='TextDetConfigGenerator')
