# This configuration prepares the ICDAR13 857 and 1015
# version, and uses ICDAR13 1015 version by default.
# You may uncomment the lines if you want to you the original version,
# which contains 1095 samples.
# You can check out the generated base config and use the 857
# version by using its corresponding config variables in your model.

data_root = 'data/icdar2013'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'Challenge2_Training_Task3_Images_GT.zip',
            save_name='ic13_textrecog_train_img_gt.zip',
            md5='6f0dbc823645968030878df7543f40a4',
            split=['train'],
            content=['image'],
            mapping=[[
                'ic13_textrecog_train_img_gt/gt.txt', 'annotations/train.txt'
            ], ['ic13_textrecog_train_img_gt', 'textrecog_imgs/train']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'Challenge2_Test_Task3_Images.zip',
            save_name='ic13_textrecog_test_img.zip',
            md5='3206778eebb3a5c5cc15c249010bf77f',
            split=['test'],
            content=['image'],
            mapping=[['ic13_textrecog_test_img', 'textrecog_imgs/test']]),
        dict(
            url='https://download.openmmlab.com/mmocr/data/1.x/recog/'
            'icdar_2013/train_labels.json',
            save_name='ic13_train_labels.json',
            md5='008fcd0056e72c4cf3064fb4d1fce81b',
            split=['train'],
            content=['annotation'],
            mapping=[['ic13_train_labels.json', 'textrecog_train.json']]),
        # Note that we offer two versions of test set annotations as follows.
        # Please choose one of them to download and comment the other. By
        # default, we use the second one.
        # 1. The original official annotation, which contains 1095 test
        # samples.
        # dict(
        #     url='https://rrc.cvc.uab.es/downloads/'
        #     'Challenge2_Test_Task3_GT.txt',
        #     save_name='ic13_textrecog_test_gt.txt',
        #     md5='2634060ed8fe6e7a4a9b8d68785835a1',
        #     split=['test'],
        #     content=['annotation'],
        #     mapping=[['ic13_textrecog_test_gt.txt', 'annotations/test.txt']]),  # noqa
        # 2. The widely-used version for academic purpose, which filters out
        # words with non-alphanumeric characters. This version contains 1015
        # test samples.
        dict(
            url='https://download.openmmlab.com/mmocr/data/1.x/recog/'
            'icdar_2013/textrecog_test_1015.json',
            save_name='textrecog_test.json',
            md5='68fdd818f63df8b93dc952478952009a',
            split=['test'],
            content=['annotation'],
        ),
        # 3. The 857 version further pruned words shorter than 3 characters.
        dict(
            url='https://download.openmmlab.com/mmocr/data/1.x/recog/'
            'icdar_2013/textrecog_test_857.json',
            save_name='textrecog_test_857.json',
            md5='3bed3985b0c51a989ad4006f6de8352b',
            split=['test'],
            content=['annotation'],
        ),
    ])

# Uncomment the data converter if you want to use the original 1095 version.
# data_converter = dict(
#     type='TextRecogDataConverter',
#     splits=['train', 'test'],
#     data_root=data_root,
#     gatherer=dict(type='mono_gather', mapping="f'{split}.txt'"),
#     parser=dict(
#         type='ICDARTxtTextRecogAnnParser', separator=', ', format='img, text'), # noqa
#     dumper=dict(type='JsonDumper'))

config_generator = dict(
    type='TextRecogConfigGenerator',
    data_root=data_root,
    test_anns=[
        dict(ann_file='textrecog_test.json'),
        dict(dataset_postfix='857', ann_file='textrecog_test_857.json')
    ])
