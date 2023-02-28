data_root = 'data/textocr'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://dl.fbaipublicfiles.com/textvqa/images/'
                'train_val_images.zip',
                save_name='textocr_textdet_train_val_img.zip',
                md5='d12dd8098899044e4ae1af34db7ecfef',
                content=['image'],
                mapping=[[
                    'textocr_textdet_train_val_img/train_images',
                    'textdet_imgs/images'
                ]]),
            dict(
                url='https://dl.fbaipublicfiles.com/textvqa/data/textocr/'
                'TextOCR_0.1_train.json',
                save_name='textocr_textdet_train.json',
                md5='0f8ba1beefd2ca4d08a4f82bcbe6cfb4',
                content=['annotation'],
                mapping=[[
                    'textocr_textdet_train.json', 'annotations/train.json'
                ]]),
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='train.json',
        img_dir='textdet_imgs/images'),
    parser=dict(type='COCOTextDetAnnParser', variant='textocr'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'))

val_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://dl.fbaipublicfiles.com/textvqa/images/'
                'train_val_images.zip',
                save_name='textocr_textdet_train_val_img.zip',
                md5='d12dd8098899044e4ae1af34db7ecfef',
                content=['image'],
                mapping=[[
                    'textocr_textdet_train_val_img/train_images',
                    'textdet_imgs/images'
                ]]),
            dict(
                url='https://dl.fbaipublicfiles.com/textvqa/data/textocr/'
                'TextOCR_0.1_val.json',
                save_name='textocr_textdet_val.json',
                md5='fb151383ea7b3c530cde9ef0d5c08347',
                content=['annotation'],
                mapping=[['textocr_textdet_val.json',
                          'annotations/val.json']]),
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='val.json',
        img_dir='textdet_imgs/images'),
    parser=dict(type='COCOTextDetAnnParser', variant='textocr'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'))
delete = ['annotations', 'textocr_textdet_train_val_img']
config_generator = dict(type='TextDetConfigGenerator')
