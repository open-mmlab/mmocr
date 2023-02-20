data_root = 'data/cocotextv2'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='http://images.cocodataset.org/zips/train2014.zip',
            save_name='cocotextv2_train_img.zip',
            md5='0da8c0bd3d6becc4dcb32757491aca88',
            split=['train', 'val'],
            content=['image'],
            mapping=[['cocotextv2_train_img/train2014',
                      'textdet_imgs/train']]),
        dict(
            url='https://github.com/bgshih/cocotext/releases/download/dl/'
            'cocotext.v2.zip',
            save_name='cocotextv2_annotation.zip',
            md5='5e39f7d6f2f11324c6451e63523c440c',
            split=['train', 'val'],
            content=['annotation'],
            mapping=[[
                'cocotextv2_annotation/cocotext.v2.json',
                'annotations/train.json'
            ]]),
    ])

data_converter = dict(
    type='TextDetDataConverter',
    splits=['train'],
    data_root=data_root,
    gatherer=dict(type='mono_gather', train_ann='train.json'),
    parser=dict(
        type='COCOTextDetAnnParser',
        variant='cocotext',
        data_root=data_root + '/textdet_imgs/train'),
    dumper=dict(type='JsonDumper'))

config_generator = dict(type='TextDetConfigGenerator', data_root=data_root)
