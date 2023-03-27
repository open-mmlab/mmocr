lang = 'fr'
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
                md5='d821ca50f37cc39ff1715632f4068ea1',
                content=['image'],
                mapping=[[f'{lang}_train/*.jpg', 'imgs/train']]),
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.train.json',
                save_name=f'{lang}_train.json',
                md5='349e7f824225bc7cc53f0c0eb8c87d3e',
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
                md5='9ccbf15816ca05e50229885b75e57e49',
                content=['image'],
                mapping=[[f'{lang}_val/*.jpg', 'imgs/test']]),
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.val.json',
                save_name=f'{lang}_val.json',
                md5='15d8a52a4eb20ea029a4aa3eaa25ef8d',
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
