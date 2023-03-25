lang = 'it'
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
                md5='c531e39f0cbc1dc74caa320ffafe5de9',
                content=['image'],
                mapping=[[f'{lang}_train/*.jpg', 'ser_imgs/train']]),
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.train.json',
                save_name=f'{lang}_train.json',
                md5='fa6afe204a6af57152627e76fe2de005',
                content=['annotation'],
                mapping=[[f'{lang}_train.json', 'annotations/train.json']])
        ]),
    gatherer=dict(
        type='MonoGatherer', ann_name='train.json', img_dir='ser_imgs/train'),
    parser=dict(type='XFUNDSERAnnParser'),
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
                md5='35446a115561d0773b7f2a0c2f32fe5c',
                content=['image'],
                mapping=[[f'{lang}_val/*.jpg', 'ser_imgs/test']]),
            dict(
                url='https://github.com/doc-analysis/XFUND/'
                f'releases/download/v1.0/{lang}.val.json',
                save_name=f'{lang}_val.json',
                md5='260d4ea447636cbca1ce1ca5fc5846d9',
                content=['annotation'],
                mapping=[[f'{lang}_val.json', 'annotations/test.json']])
        ]),
    gatherer=dict(
        type='MonoGatherer', ann_name='test.json', img_dir='ser_imgs/test'),
    parser=dict(type='XFUNDSERAnnParser'),
    packer=dict(type='SERPacker'),
    dumper=dict(type='JsonDumper'),
)

delete = ['annotations'] + [f'{lang}_{split}' for split in ['train', 'val']]
config_generator = dict(type='SERConfigGenerator')
