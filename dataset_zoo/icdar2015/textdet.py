data_root = 'data/icdar2015'
cache_path = 'data/cache'
train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',
                save_name='ic15_textdet_train_img.zip',
                md5='c51cbace155dcc4d98c8dd19d378f30d',
                content=['image'],
                mapping=[['ic15_textdet_train_img', 'textdet_imgs/train']]),
            dict(
                url='https://rrc.cvc.uab.es/downloads/'
                'ch4_training_localization_transcription_gt.zip',
                save_name='ic15_textdet_train_gt.zip',
                md5='3bfaf1988960909014f7987d2343060b',
                content=['annotation'],
                mapping=[['ic15_textdet_train_gt', 'annotations/train']]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg', '.JPG'],
        rule=[r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt']),
    parser=dict(type='ICDARTxtTextDetAnnParser', encoding='utf-8-sig'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_test_images.zip',
                save_name='ic15_textdet_test_img.zip',
                md5='97e4c1ddcf074ffcc75feff2b63c35dd',
                content=['image'],
                mapping=[['ic15_textdet_test_img', 'textdet_imgs/test']]),
            dict(
                url='https://rrc.cvc.uab.es/downloads/'
                'Challenge4_Test_Task4_GT.zip',
                save_name='ic15_textdet_test_gt.zip',
                md5='8bce173b06d164b98c357b0eb96ef430',
                content=['annotation'],
                mapping=[['ic15_textdet_test_gt', 'annotations/test']]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg', '.JPG'],
        rule=[r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt']),
    parser=dict(type='ICDARTxtTextDetAnnParser', encoding='utf-8-sig'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

config_generator = dict(type='TextDetConfigGenerator')
delete = ['annotations', 'ic15_textdet_test_img', 'ic15_textdet_train_img']
