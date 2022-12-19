data_root = 'data/sroie'
cache_path = 'data/cache'

data_obtainer = dict(
 type='NaiveDataObtainer',
 cache_path=cache_path,
 data_root=data_root,
 files=[
        dict(
            url='',
            save_name='0325updated.task1train\(626p\).zip',
            md5='',
            split=['train'],
            content=['image', 'annotation'],
            mapping=[['0325updated.task1train\(626p\)/*.jpg', 'textdet_imgs/train/*.jpg']
                     ['0325updated.task1train\(626p\)/*.txt', 'annotations/train/*.txt']]),
        dict(
            url='',
            save_name='task1_2_test\(361p\).zip',
            md5='',
            split=['test'],
            content=['image'],
            mapping=[['task1_2_test\(361p\)', 'textdet_imgs/test']]),
        dict(
            url='',
            save_name='text.task1\&2-test（361p\).zip',
            md5='',
            split=['test'],
            content=['image'],
            mapping=[['text.task1_2-test（361p\)', 'annotations/test']]),
    ])


data_converter = dict(
    type='TextDetDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='pair_gather', 
        suffixes=['.jpg'],
        rule=[r'(\w+)\.jpg', r'\1.txt']),
    parser=dict(type='ICDARTxtTextDetAnnParser', encoding='utf-8-sig'),
    dumper=dict(type='JsonDumper'),
    # delete=['0325updated.task1train\(626p\)', 'task1_2_test\(361p\)', 
    #         'text.task1\&2-test（361p\)']
)

config_generator = dict(type='TextDetConfigGenerator', data_root=data_root)
