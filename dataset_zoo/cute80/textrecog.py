# The official annotations of CUTE80 has some typos, and we have provided
# the fixed version as done in
# https://github.com/clovaai/deep-text-recognition-benchmark by default.
# If you want to use the original version, please comment out the following
# lines: L10-L31, and uncomment L33-L63

data_root = 'data/cute80'
cache_path = 'data/cache'

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        data_root=data_root,
        files=[
            dict(
                url='https://download.openmmlab.com/mmocr/data/mixture/ct80/'
                'timage.tar.gz',
                save_name='ct80.tar.gz',
                md5='9f3b1fe0e76f1fdfc70de3a365603d5e',
                split=['test'],
                content=['image'],
                mapping=[['ct80/timage', 'textrecog_imgs/test']]),
            dict(
                url='https://download.openmmlab.com/mmocr/data/1.x/recog/ct80/'
                'textrecog_test.json',
                save_name='textrecog_test.json',
                md5='9c5c79d843b900325e7fd453b318cad9',
                split=['test'],
                content=['annotation'])
        ]))

# test_preparer = dict(
#     obtainer=dict(
#         type='NaiveDataObtainer',
#         cache_path=cache_path,
#         data_root=data_root,
#         files=[
#             dict(
#                 url='https://download.openmmlab.com/mmocr/data/mixture/ct80/'
#                 'timage.tar.gz',
#                 save_name='ct80.tar.gz',
#                 md5='9f3b1fe0e76f1fdfc70de3a365603d5e',
#                 split=['test'],
#                 content=['image'],
#                 mapping=[['ct80/timage', 'textrecog_imgs/test']]),
#             dict(
#                 url='https://download.openmmlab.com/mmocr/data/mixture/ct80/'
#                 'test_label.txt',
#                 save_name='ct80_test.txt',
#                 md5='f679dec62916d3268aff9cd81990d260',
#                 split=['test'],
#                 content=['annotation'],
#                 mapping=[['ct80_test.txt', 'annotations/test.txt']])
#         ]),
#     gatherer=dict(type='MonoGatherer', ann_name='test.txt'),
#     parser=dict(
#         type='ICDARTxtTextRecogAnnParser',
#         separator=' ',
#         format='img text ignore1 ignore2'),
#     packer=dict(type='TextRecogPacker'),
#     dumper=dict(type='JsonDumper'),
# )
delete = ['ct80']
config_generator = dict(
    type='TextRecogConfigGenerator', data_root=data_root, train_anns=None)
