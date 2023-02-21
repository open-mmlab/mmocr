data_root = 'data/svtp'
cache_path = 'data/cache'

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://download.openmmlab.com/mmocr/data/svtp.zip',
                save_name='svtp.zip',
                md5='4232b46c81ba99eea6d057dcb06b8f75',
                split=['test'],
                content=['image', 'annotation'],
                mapping=[['svtp/par1', 'textrecog_imgs/test'],
                         ['svtp/gt.txt', 'annotations/test.txt']]),
        ]),
    gatherer=dict(type='MonoGatherer', ann_name='test.txt'),
    parser=dict(
        type='ICDARTxtTextRecogAnnParser', separator=' ', format='img text'),
    packer=dict(type='TextRecogPacker'),
    dumper=dict(type='JsonDumper'))

config_generator = dict(type='TextRecogConfigGenerator')
