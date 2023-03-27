lang = 'de'
data_root = f'data/xfund/{lang}'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.train.zip',
                save_name=f'{lang}_train.zip',
                md5='8c9f949952d227290e22f736cdbe4d29',
                content=['image'],
                mapping=[[f'{lang}_train/*.jpg', 'imgs/train']]),
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.train.json',
                save_name=f'{lang}_train.json',
                md5='3e4b95c7da893bf5a91018445c83ccdd',
                content=['annotation'],
                mapping=[[f'{lang}_train.json', 'annotations/train.json']])
        ]),
    gatherer=dict(
        type='MonoGatherer', ann_name='train.json', img_dir='imgs/train'),
    parser=dict(type='XFUNDAnnParser'),
    packer=dict(type='SERPacker'),
    dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.val.zip',
                save_name=f'{lang}_val.zip',
                md5='d13d12278d585214183c3cfb949b0e59',
                content=['image'],
                mapping=[[f'{lang}_val/*.jpg', 'imgs/test']]),
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.val.json',
                save_name=f'{lang}_val.json',
                md5='8eaf742f2d19b17f5c0e72da5c7761ef',
                content=['annotation'],
                mapping=[[f'{lang}_val.json', 'annotations/test.json']])
        ]),
    gatherer=dict(
        type='MonoGatherer', ann_name='test.json', img_dir='imgs/test'),
    parser=dict(type='XFUNDAnnParser'),
    packer=dict(type='SERPacker'),
    dumper=dict(type='JsonDumper'),
)

delete = ['annotations'] + [f'{lang}_{split}' for split in ['train', 'val']]
config_generator = dict(type='SERConfigGenerator')
