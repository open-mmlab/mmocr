data_root = 'data/icdar2015'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',
            save_name='ic15_textdet_train_img.zip',
            md5='c51cbace155dcc4d98c8dd19d378f30d',
            split=['train'],
            content=['image'],
            mapping=[['ic15_textdet_train_img', 'textdet_imgs/train']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/ch4_test_images.zip',
            save_name='ic15_textdet_test_img.zip',
            md5='97e4c1ddcf074ffcc75feff2b63c35dd',
            split=['test'],
            content=['image'],
            mapping=[['ic15_textdet_test_img', 'textdet_imgs/test']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'ch4_training_localization_transcription_gt.zip',
            save_name='ic15_textdet_train_gt.zip',
            md5='3bfaf1988960909014f7987d2343060b',
            split=['train'],
            content=['annotation'],
            mapping=[['ic15_textdet_train_gt', 'annotations/train']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'Challenge4_Test_Task4_GT.zip',
            save_name='ic15_textdet_test_gt.zip',
            md5='8bce173b06d164b98c357b0eb96ef430',
            split=['test'],
            content=['annotation'],
            mapping=[['ic15_textdet_test_gt', 'annotations/test']]),
    ])

data_converter = dict(
    type='TextDetDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='pair_gather',
        suffixes=['.jpg', '.JPG'],
        rule=[r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt']),
    parser=dict(type='ICDARTxtTextDetAnnParser'),
    dumper=dict(type='JsonDumper'),
    delete=['annotations', 'ic15_textdet_test_img', 'ic15_textdet_train_img'])
