data_root = 'data/cocotextv2'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='http://images.cocodataset.org/zips/train2014.zip',
                save_name='cocotextv2_train_img.zip',
                md5='0da8c0bd3d6becc4dcb32757491aca88',
                content=['image'],
                mapping=[[
                    'cocotextv2_train_img/train2014', 'textdet_imgs/imgs'
                ]]),
            dict(
                url='https://github.com/bgshih/cocotext/releases/download/dl/'
                'cocotext.v2.zip',
                save_name='cocotextv2_annotation.zip',
                md5='5e39f7d6f2f11324c6451e63523c440c',
                content=['annotation'],
                mapping=[[
                    'cocotextv2_annotation/cocotext.v2.json',
                    'annotations/train.json'
                ]]),
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='train.json',
        img_dir='textdet_imgs/imgs'),
    parser=dict(type='COCOTextDetAnnParser', variant='cocotext'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'))

val_preparer = train_preparer

delete = ['annotations', 'cocotextv2_annotation', 'cocotextv2_train_img']
config_generator = dict(type='TextDetConfigGenerator')
