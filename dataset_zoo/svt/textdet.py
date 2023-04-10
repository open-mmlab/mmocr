data_root = 'data/svt'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='http://www.iapr-tc11.org/dataset/SVT/svt.zip',
                save_name='svt.zip',
                md5='42d19160010d990ae6223b14f45eff88',
                content=['image', 'annotations'],
                mapping=[['svt/svt1/train.xml', 'annotations/train.xml'],
                         ['svt/svt1/img', 'textdet_imgs/img']]),
        ]),
    gatherer=dict(
        type='MonoGatherer', ann_name='train.xml', img_dir='textdet_imgs/img'),
    parser=dict(type='SVTTextDetAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='http://www.iapr-tc11.org/dataset/SVT/svt.zip',
                save_name='svt.zip',
                md5='42d19160010d990ae6223b14f45eff88',
                content=['image', 'annotations'],
                mapping=[['svt/svt1/test.xml', 'annotations/test.xml'],
                         ['svt/svt1/img', 'textdet_imgs/img']]),
        ]),
    gatherer=dict(
        type='MonoGatherer', ann_name='test.xml', img_dir='textdet_imgs/img'),
    parser=dict(type='SVTTextDetAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)
delete = ['annotations', 'svt']
config_generator = dict(type='TextDetConfigGenerator')
