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
            mapping=[['ic13_textrecog_train_img_gt', 'textrecog_imgs/train']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'Challenge2_Test_Task3_Images.zip',
            save_name='ic13_textrecog_test_img.zip',
            md5='3206778eebb3a5c5cc15c249010bf77f',
            split=['test'],
            content=['image'],
            mapping=[['ic13_textrecog_test_img', 'textrecog_imgs/test']]),
        dict(
            url='https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/'
            'train_label.txt',
            save_name='ic13_textrecog_train_gt.txt',
            md5='854d4af1237875b2baad504aa70d3864',
            split=['train'],
            content=['annotation'],
            mapping=[['ic13_textrecog_train_gt.txt',
                      'annotations/train.txt']]),
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
        #     mapping=[['ic13_textrecog_test_gt.txt', 'annotations/test.txt']])
        # 2. The widely-used version for academic purpose, which filters out
        # words with non-alphanumeric characters. This version contains 1015
        # test samples.
        dict(
            url='https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/'
            'test_label_1015.txt',
            save_name='ic13_textrecog_test_gt.txt',
            md5='66a4a1c5332e6899c63e927ba890f68c',
            split=['test'],
            content=['annotation'],
            mapping=[['ic13_textrecog_test_gt.txt', 'annotations/test.txt']])
    ])

data_converter = dict(
    type='TextRecogDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(type='mono_gather', mapping="f'{split}.txt'"),
    parser=dict(
        type='ICDARTxtTextRecogAnnParser', separator=' ', format='img text'),
    dumper=dict(type='JsonDumper'))
