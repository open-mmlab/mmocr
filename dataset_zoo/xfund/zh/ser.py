lang = 'zh'
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
                md5='a4ce16d1c1a8554a8b1e00907cff3b4b',
                content=['image'],
                mapping=[[f'{lang}_train/*.jpg', 'imgs/train']]),
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.train.json',
                save_name=f'{lang}_train.json',
                md5='af1afd5e935cccd3a105de6c12eb4c31',
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
                md5='f84c2651e350f5b394585207a43d06e4',
                content=['image'],
                mapping=[[f'{lang}_val/*.jpg', 'imgs/test']]),
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.val.json',
                save_name=f'{lang}_val.json',
                md5='c243c35d1685a16435c8b281a445005c',
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
config_generator = dict(type='XFUNDSERConfigGenerator')
