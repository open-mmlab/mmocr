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
        ]),
    gatherer=dict(type='MonoGatherer', ann_name='annotation.txt'),
    parser=dict(
        type='ICDARTxtTextRecogAnnParser',
        separator=' ',
        format='img text',
        remove_strs=None),
    packer=dict(type='TextRecogPacker'),
    dumper=dict(type='JsonDumper'),
)

delete = ['mjsynth', 'annotations']

config_generator = dict(
    type='TextRecogConfigGenerator', data_root=data_root, test_anns=None)
