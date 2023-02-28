data_root = 'data/sroie'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://download.openmmlab.com/mmocr/data/'
                'sroie/0325updated.task1train(626p).zip',
                save_name='0325updated.task1train(626p).zip',
                md5='16137490f6865caac75772b9111d348c',
                content=['image', 'annotation'],
                mapping=[[
                    '0325updated/0325updated.task1train(626p)/*.jpg',
                    'textdet_imgs/train'
                ],
                         [
                             '0325updated/0325updated.task1train(626p)/*.txt',
                             'annotations/train'
                         ]])
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg'],
        rule=[r'X(\d+)\.([jJ][pP][gG])', r'X\1.txt']),
    parser=dict(type='SROIETextDetAnnParser', encoding='utf-8-sig'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://download.openmmlab.com/mmocr/data/'
                'sroie/task1&2_test(361p).zip',
                save_name='task1&2_test(361p).zip',
                md5='1bde54705db0995c57a6e34cce437fea',
                content=['image'],
                mapping=[[
                    'task1&2_test(361p)/fulltext_test(361p)',
                    'textdet_imgs/test'
                ]]),
            dict(
                url='https://download.openmmlab.com/mmocr/data/sroie/text.zip',
                save_name='text.zip',
                md5='8c534653f252ff4d3943fa27a956a74b',
                content=['annotation'],
                mapping=[['text', 'annotations/test']]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg'],
        rule=[r'X(\d+)\.([jJ][pP][gG])', r'X\1.txt']),
    parser=dict(type='SROIETextDetAnnParser', encoding='utf-8-sig'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)
delete = ['text', 'task1&2_test(361p)', '0325updated', 'annotations']
config_generator = dict(type='TextDetConfigGenerator', data_root=data_root)
