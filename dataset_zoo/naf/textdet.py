data_root = 'data/naf'
cache_path = 'data/cache'

obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    files=[
        dict(
            url='https://github.com/herobd/NAF_dataset/releases/'
            'download/v1.0/labeled_images.tar.gz',
            save_name='naf_image.tar.gz',
            md5='6521cdc25c313a1f2928a16a77ad8f29',
            split=['train', 'test', 'val'],
            content=['image'],
            mapping=[['naf_image/labeled_images', 'temp_images/']]),
        dict(
            url='https://github.com/herobd/NAF_dataset/archive/'
            'refs/heads/master.zip',
            save_name='naf_anno.zip',
            md5='abf5af6266cc527d772231751bc884b3',
            split=['train', 'test', 'val'],
            content=['annotation'],
            mapping=[
                [
                    'naf_anno/NAF_dataset-master/groups/**/*.json',
                    'annotations/'
                ],
                [
                    'naf_anno/NAF_dataset-master/train_valid_test_split.json',
                    'data_split.json'
                ]
            ]),
    ])

prepare_train_data = dict(
    obtainer=obtainer,
    gatherer=dict(type='NAFGatherer'),
    parser=dict(type='NAFAnnParser', det=True),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

prepare_test_data = prepare_train_data

prepare_val_data = prepare_train_data

config_generator = dict(
    type='TextDetConfigGenerator',
    val_anns=[dict(ann_file='textdet_val.json', dataset_postfix='')])
