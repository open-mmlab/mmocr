data_root = './data/wildreceipt'
cache_path = './data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://download.openmmlab.com/mmocr/data/wildreceipt.tar',
            save_name='wildreceipt.tar',
            md5='2a2c4a1b4777fb4fe185011e17ad46ae',
            split=['train', 'test'],
            content=['image', 'annotation'],
            mapping=[
                ['wildreceipt/wildreceipt/class_list.txt', 'class_list.txt'],
                ['wildreceipt/wildreceipt/dict.txt', 'dict.txt'],
                ['wildreceipt/wildreceipt/test.txt', 'test.txt'],
                ['wildreceipt/wildreceipt/train.txt', 'train.txt'],
                ['wildreceipt/wildreceipt/image_files', 'image_files'],
            ]),
    ])

data_converter = dict(
    type='WildReceiptConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='mono_gather', mapping="f'{split}.txt'", ann_path=data_root),
    parser=dict(type='WildreceiptKIEAnnParser', data_root=data_root),
    dumper=dict(type='WildreceiptOpensetDumper'),
    delete=['wildreceipt'])
