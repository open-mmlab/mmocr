data_root = 'data/icdar2015'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'ch4_training_word_images_gt.zip',
            save_name='ic15_textrecog_train_img_gt.zip',
            md5='600caf8c6a64a3dcf638839820edcca9',
            split=['train'],
            content=['image', 'annotation'],
            mapping=[[
                'ic15_textrecog_train_img_gt/gt.txt', 'annotations/train.txt'
            ], ['ic15_textrecog_train_img_gt', 'crops/train']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/ch4_test_word_images_gt.zip',
            save_name='ic15_textrecog_test_img.zip',
            md5='d7a71585f4cc69f89edbe534e7706d5d',
            split=['test'],
            content=['image'],
            mapping=[['ic15_textrecog_test_img', 'crops/test']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'Challenge4_Test_Task3_GT.txt',
            save_name='ic15_textrecog_test_gt.txt',
            md5='d7a71585f4cc69f89edbe534e7706d5d',
            split=['test'],
            content=['annotation'],
            mapping=[['ic15_textrecog_test_gt.txt', 'annotations/test.txt']])
    ])

data_converter = dict(
    type='TextRecogDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(type='mono_gather', mapping="f'{split}.txt'"),
    parser=dict(type='ICDAR2015TextRecogAnnParser'),
    dumper=dict(type='JsonDumper'))
