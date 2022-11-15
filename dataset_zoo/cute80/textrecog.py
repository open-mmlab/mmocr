data_root = 'data/cute80'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://download.openmmlab.com/mmocr/data/mixture/ct80/'
            'timage.tar.gz',
            save_name='ct80.tar.gz',
            md5='9f3b1fe0e76f1fdfc70de3a365603d5e',
            split=['test'],
            content=['image'],
            mapping=[['ct80/timage', 'textrecog_imgs/test']]),
        dict(
            url='https://download.openmmlab.com/mmocr/data/mixture/ct80/'
            'test_label.txt',
            save_name='ct80_test.txt',
            md5='f679dec62916d3268aff9cd81990d260',
            split=['test'],
            content=['annotation'],
            mapping=[['ct80_test.txt', 'annotations/test.txt']])
    ])

data_converter = dict(
    type='TextRecogDataConverter',
    splits=['test'],
    data_root=data_root,
    gatherer=dict(type='mono_gather', mapping="f'{split}.txt'"),
    parser=dict(
        type='ICDARTxtTextRecogAnnParser',
        separator=' ',
        format='img text ignore1 ignore2'),
    dumper=dict(type='JsonDumper'))
