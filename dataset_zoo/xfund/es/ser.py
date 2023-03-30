lang = 'es'
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
                md5='0ff89032bc6cb2e7ccba062c71944d03',
                content=['image'],
                mapping=[[f'{lang}_train/*.jpg', 'imgs/train']]),
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.train.json',
                save_name=f'{lang}_train.json',
                md5='b40b43f276c7deaaaa5923d035da2820',
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
                md5='efad9fb11ee3036bef003b6364a79ac0',
                content=['image'],
                mapping=[[f'{lang}_val/*.jpg', 'imgs/test']]),
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.val.json',
                save_name=f'{lang}_val.json',
                md5='96ffc2057049ba2826a005825b3e7f0d',
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
