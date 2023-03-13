data_root = 'data/synthtext'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='magnet:?xt=urn:btih:2dba9518166cbd141534cbf381aa3e99a08'
                '7e83c&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&t'
                'r=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2F'
                'tracker.opentrackr.org%3A1337%2Fannounce',
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
