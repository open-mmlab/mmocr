data_root = 'data/textocr'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://dl.fbaipublicfiles.com/textvqa/images/'
            'train_val_images.zip',
            save_name='textocr_textdet_train_val_img.zip',
            md5='d12dd8098899044e4ae1af34db7ecfef',
            split=['train', 'val'],
            content=['image'],
            mapping=[[
                'textocr_textdet_train_val_img/train_images',
                'textdet_imgs/train'
            ]]),
        dict(
            url='https://dl.fbaipublicfiles.com/textvqa/data/textocr/'
            'TextOCR_0.1_train.json',
            save_name='textocr_textdet_train.json',
            md5='0f8ba1beefd2ca4d08a4f82bcbe6cfb4',
            split=['train'],
            content=['annotation'],
            mapping=[['textocr_textdet_train.json',
                      'annotations/train.json']]),
        dict(
            url='https://dl.fbaipublicfiles.com/textvqa/data/textocr/'
            'TextOCR_0.1_val.json',
            save_name='textocr_textdet_val.json',
            md5='fb151383ea7b3c530cde9ef0d5c08347',
            split=['val'],
            content=['annotation'],
            mapping=[['textocr_textdet_val.json', 'annotations/val.json']]),
    ])

data_converter = dict(
    type='TextDetDataConverter',
    splits=['train', 'val'],
    data_root=data_root,
    gatherer=dict(type='mono_gather', mapping="f'{split}.json'"),
    parser=dict(
        type='COCOTextDetAnnParser',
        variant='textocr',
        data_root=data_root + '/textdet_imgs/'),
    dumper=dict(type='JsonDumper'),
    delete=['annotations', 'textocr_textdet_train_val_img'])
