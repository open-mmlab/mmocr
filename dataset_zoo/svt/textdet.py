data_root = 'data/svt'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='http://www.iapr-tc11.org/dataset/SVT/svt.zip',
            save_name='svt.zip',
            md5='42d19160010d990ae6223b14f45eff88',
            split=['train', 'test'],
            content=['image', 'annotations'],
            mapping=[['svt/svt1/train.xml', 'annotations/train.xml'],
                     ['svt/svt1/test.xml', 'annotations/test.xml'],
                     ['svt/svt1/img', 'textdet_imgs/img']]),
    ])

data_converter = dict(
    type='TextDetDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='mono_gather', train_ann='train.xml', test_ann='test.xml'),
    parser=dict(type='SVTTextDetAnnParser', data_root=data_root),
    dumper=dict(type='JsonDumper'),
    delete=['annotations', 'svt'])

config_generator = dict(type='TextDetConfigGenerator', data_root=data_root)
