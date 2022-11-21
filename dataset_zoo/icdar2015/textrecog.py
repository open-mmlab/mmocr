# This configuration prepares the ICDAR15 1811 and 2077
# version, and uses ICDAR15 2077 version by default.
# Read https://arxiv.org/pdf/1904.01906.pdf for more info.
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
            ], ['ic15_textrecog_train_img_gt', 'textrecog_imgs/train']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/ch4_test_word_images_gt.zip',
            save_name='ic15_textrecog_test_img.zip',
            md5='d7a71585f4cc69f89edbe534e7706d5d',
            split=['test'],
            content=['image'],
            mapping=[['ic15_textrecog_test_img', 'textrecog_imgs/test']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'Challenge4_Test_Task3_GT.txt',
            save_name='ic15_textrecog_test_gt.txt',
            md5='d7a71585f4cc69f89edbe534e7706d5d',
            split=['test'],
            content=['annotation'],
            mapping=[['ic15_textrecog_test_gt.txt', 'annotations/test.txt']]),
        # 3. The 1811 version discards non-alphanumeric character images and
        # some extremely rotated, perspective-shifted, and curved images for
        # evaluation
        dict(
            url='https://download.openmmlab.com/mmocr/data/1.x/recog/'
            'icdar_2015/textrecog_test_1811.json',
            save_name='textrecog_test_1811.json',
            md5='8d218ef1c37540ea959e22eeabc79ae4',
            split=['test'],
            content=['annotation'],
        ),
    ])

data_converter = dict(
    type='TextRecogDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(type='mono_gather', mapping="f'{split}.txt'"),
    parser=dict(type='ICDARTxtTextRecogAnnParser', encoding='utf-8-sig'),
    dumper=dict(type='JsonDumper'))

config_generator = dict(
    type='TextRecogConfigGenerator',
    data_root=data_root,
    test_anns=[
        dict(ann_file='textrecog_test.json'),
        dict(dataset_postfix='1811', ann_file='textrecog_test_1811.json')
    ])
