data_root = 'data/iiit5k'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/'
            'IIIT5K-Word_V3.0.tar.gz',
            save_name='IIIT5K.tar.gz',
            md5='56781bc327d22066aa1c239ee788fd46',
            split=['test', 'train'],
            content=['image'],
            mapping=[['IIIT5K/IIIT5K/test', 'textrecog_imgs/test'],
                     ['IIIT5K/IIIT5K/train', 'textrecog_imgs/train']]),
        dict(
            url='https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/'
            'test_label.txt',
            save_name='iiit5k_test.txt',
            md5='82ecfa34a28d59284d1914dc906f5380',
            split=['test'],
            content=['annotation'],
            mapping=[['iiit5k_test.txt', 'annotations/test.txt']]),
        dict(
            url='https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/'
            'train_label.txt',
            save_name='iiit5k_train.txt',
            md5='f4731ce1eadc259532c2834266e5126d',
            split=['train'],
            content=['annotation'],
            mapping=[['iiit5k_train.txt', 'annotations/train.txt']]),
    ])

data_converter = dict(
    type='TextRecogDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='MonoGatherer',
        train_ann_name='train.txt',
        test_ann_name='test.txt'),
    parser=dict(
        type='ICDARTxtTextRecogAnnParser',
        encoding='utf-8',
        separator=' ',
        format='img text'),
    dumper=dict(type='JsonDumper'),
    delete=['annotations', 'IIIT5K'])

config_generator = dict(type='TextRecogConfigGenerator', data_root=data_root)
