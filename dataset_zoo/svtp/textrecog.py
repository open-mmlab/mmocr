data_root = 'data/svtp'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://download.openmmlab.com/mmocr/data/svtp.zip',
            save_name='svtp.zip',
            md5='4232b46c81ba99eea6d057dcb06b8f75',
            split=['test'],
            content=['image', 'annotation'],
            mapping=[['svtp/par1', 'textrecog_imgs/test'],
                     ['svtp/gt.txt', 'annotations/test.txt']]),
    ])

data_converter = dict(
    type='TextRecogDataConverter',
    splits=['test'],
    data_root=data_root,
    gatherer=dict(type='mono_gather', mapping="f'{split}.txt'"),
    parser=dict(
        type='ICDARTxtTextRecogAnnParser', separator=' ', format='img text'),
    dumper=dict(type='JsonDumper'),
    delete=['svtp', 'annotations'])
