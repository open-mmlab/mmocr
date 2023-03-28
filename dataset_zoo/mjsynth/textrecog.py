data_root = 'data/mjsynth'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://thor.robots.ox.ac.uk/~vgg/data/text/'
                'mjsynth.tar.gz',
                save_name='mjsynth.tar.gz',
                md5='7bf2b60ad935eaf64e5b606f782d68e5',
                split=['train'],
                content=['image', 'annotation'],
                mapping=[
                    [
                        'mjsynth/mnt/ramdisk/max/90kDICT32px/*/',
                        'textrecog_imgs/train/'
                    ],
                    [
                        'mjsynth/mnt/ramdisk/max/90kDICT32px/annotation.txt',
                        'annotations/annotation.txt'
                    ]
                ]),
            dict(
                url='https://download.openmmlab.com/mmocr/data/1.x/recog/'
                'Syn90k/subset_textrecog_train.json',
                save_name='subset_textrecog_train.json',
                md5='ba958d87bb170980f39e194180c15b9e',
                split=['train'],
                content=['annotation'])
        ]),
    gatherer=dict(type='MonoGatherer', ann_name='annotation.txt'),
    parser=dict(
        type='MJSynthAnnParser',
        separator=' ',
        format='img num',
        remove_strs=None),
    packer=dict(type='TextRecogPacker'),
    dumper=dict(type='JsonDumper'),
)

delete = ['mjsynth', 'annotations']

config_generator = dict(
    type='TextRecogConfigGenerator',
    data_root=data_root,
    train_anns=[
        dict(ann_file='textrecog_train.json', dataset_postfix=''),
        dict(ann_file='subset_textrecog_train.json', dataset_postfix='sub'),
    ],
    test_anns=None)
