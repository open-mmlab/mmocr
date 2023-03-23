data_root = 'data/xfund'
cache_path = 'data/cache'
langs = ['zh']

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        data_root=data_root,
        files=[dict(url=f'https://github.com/doc-analysis/XFUND/releases/tag/v1.0/{lang}.train.zip',
                    save_name=f'{lang}_train.zip',
                    md5='a4ce16d1c1a8554a8b1e00907cff3b4b',
                    content=['image'],
                    mapping=[
                        [
                            f'{lang}_train/*.jpg',
                            f'ner_imgs/{lang}/train'
                        ]
                    ]) for lang in langs] +
              [dict(url=f'https://github.com/doc-analysis/XFUND/releases/tag/v1.0/{lang}.train.json',
                    save_name=f'{lang}_train.json',
                    md5='af1afd5e935cccd3a105de6c12eb4c31',
                    content=['annotation'],
                    mapping=[
                        [
                            f'{lang}_train.json',
                            f'annotations/{lang}/ner_train.json'
                        ]
                    ]) for lang in langs]
            ),
    # gatherer=dict(
    #     type='PairGatherer',
    #     img_suffixes=['.png'],
    #     rule=[r'(\w+)\.png', r'\1.json']),
    # parser=dict(type='FUNSDTextDetAnnParser'),
    # packer=dict(type='TextDetPacker'),
    # dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[dict(url=f'https://github.com/doc-analysis/XFUND/releases/tag/v1.0/{lang}.val.zip',
                    save_name=f'{lang}_val.zip',
                    md5='f84c2651e350f5b394585207a43d06e4',
                    content=['image'],
                    mapping=[
                        [
                            f'{lang}_val/*.jpg',
                            f'ner_imgs/{lang}/test'
                        ]
                    ]) for lang in langs] +
              [dict(url=f'https://github.com/doc-analysis/XFUND/releases/tag/v1.0/{lang}.val.json',
                    save_name=f'{lang}_val.json',
                    md5='c243c35d1685a16435c8b281a445005c',
                    content=['annotation'],
                    mapping=[
                        [
                            f'{lang}_val.json',
                            f'annotations/{lang}/ner_test.json'
                        ]
                    ]) for lang in langs]
            ),
    # gatherer=dict(
    #     type='PairGatherer',
    #     img_suffixes=['.png'],
    #     rule=[r'(\w+)\.png', r'\1.json']),
    # parser=dict(type='FUNSDTextDetAnnParser'),
    # packer=dict(type='TextDetPacker'),
    # dumper=dict(type='JsonDumper'),
)
delete = ['annotations', 'funsd']
config_generator = dict(type='TextDetConfigGenerator')
