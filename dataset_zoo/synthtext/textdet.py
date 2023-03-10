data_root = 'data/synthtext'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://thor.robots.ox.ac.uk/~vgg/data/scenetext/'
                'SynthText-v1.zip',
                save_name='SynthText-v1.zip',
                md5='d588045cc6173afd91c25c1e089b36f3',
                split=['train'],
                content=['image', 'annotation'],
                mapping=[['SynthText-v1/SynthText/*', 'textdet_imgs/train/'],
                         ['textdet_imgs/train/gt.mat', 'annotations/gt.mat']]),
        ]),
    gatherer=dict(type='MonoGatherer', ann_name='gt.mat'),
    parser=dict(type='SynthTextTextDetAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

config_generator = dict(
    type='TextDetConfigGenerator', data_root=data_root, test_anns=None)
