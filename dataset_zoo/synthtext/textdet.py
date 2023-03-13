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
                save_name='SynthText.zip',
                md5='8ae0309c80ff882f9d6ba5ea62cdb556',
                split=['train'],
                content=['image', 'annotation'],
                mapping=[['SynthText/SynthText/*', 'textdet_imgs/train/'],
                         ['textdet_imgs/train/gt.mat', 'annotations/gt.mat']]),
        ]),
    gatherer=dict(type='MonoGatherer', ann_name='gt.mat'),
    parser=dict(type='SynthTextTextDetAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

delete = ['SynthText']

config_generator = dict(
    type='TextDetConfigGenerator', data_root=data_root, test_anns=None)
