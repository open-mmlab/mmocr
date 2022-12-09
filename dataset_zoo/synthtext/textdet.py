data_root = 'data/synthtext'
cache_path = 'data/cache'

# data_obtainer = dict(
#     type='NaiveDataObtainer',
#     cache_path=cache_path,
#     data_root=data_root,
#     files=[
#         dict(
#             url='https://thor.robots.ox.ac.uk/~vgg/data/scenetext/'
#             'SynthText-v1.zip',
#             save_name='SynthText-v1.zip',
#             md5='d588045cc6173afd91c25c1e089b36f3',
#             split=['train'],
#             content=['image', 'annotation'],
#             mapping=[[f'SynthText/{i}', f'textdet_imgs/train/{i}']
#                      for i in range(1, 201)] +
#             [['SynthText/gt.mat', 'annotations/']]),
#     ])

data_converter = dict(
    type='TextDetDataConverter',
    splits=['train'],
    data_root=data_root,
    gatherer=dict(type='mono_gather', train_ann='gt.mat'),
    parser=dict(type='SynthTextTextDetAnnParser'),
    dumper=dict(type='JsonDumper'),
)
# delete=['annotations', 'ic15_textdet_test_img', 'ic15_textdet_train_img'])

config_generator = dict(
    type='TextDetConfigGenerator', data_root=data_root, test_anns=None)
