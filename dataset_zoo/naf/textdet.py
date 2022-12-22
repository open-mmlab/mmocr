data_root = 'data/naf'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NAFDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
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
                ['naf_anno/NAF_dataset-master/groups', 'annotations/'],
                [
                    'naf_anno/NAF_dataset-master/train_valid_test_split.json',
                    'data_split.json'
                ]
            ]),
    ])

data_converter = dict(
    type='TextDetDataConverter',
    splits=['train', 'test', 'val'],
    data_root=data_root,
    gatherer=dict(type='naf_gather'),
    parser=dict(type='NAFTextDetAnnParser', data_root=data_root, det=True),
    delete=['temp_images', 'data_split.json', 'annotations', 'naf_anno'],
    dumper=dict(type='JsonDumper'),
    nproc=1)

config_generator = dict(
    type='TextDetConfigGenerator',
    data_root=data_root,
    val_anns=[dict(ann_file='textdet_val.json', dataset_postfix='')])
