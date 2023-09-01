data_root = 'data/detext'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/'
                'ch9_training_images.zip',
                save_name='detext_textdet_train_img.zip',
                md5='e07161d6af1ef2f81f9ba0d2f904e377',
                content=['image'],
                mapping=[['detext_textdet_train_img', 'textdet_imgs/train']]),
            dict(
                url='https://rrc.cvc.uab.es/downloads/'
                'ch9_training_localization_transcription_gt.zip',
                save_name='detext_textdet_train_gt.zip',
                md5='ae4dfe155e61dcfeadd80f6b0fd15626',
                content=['annotation'],
                mapping=[['detext_textdet_train_gt', 'annotations/train']]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg'],
        rule=[r'(\w+)\.jpg', r'gt_\1.txt']),
    parser=dict(type='DetextDetAnnParser', encoding='utf-8-sig'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/'
                'ch9_validation_images.zip',
                save_name='detext_textdet_test_img.zip',
                md5='c6ffe0abe6f2d7b4d70e6883257308e0',
                content=['image'],
                mapping=[['detext_textdet_test_img', 'textdet_imgs/test']]),
            dict(
                url='https://rrc.cvc.uab.es/downloads/'
                'ch9_validation_localization_transcription_gt.zip',
                save_name='detext_textdet_test_gt.zip',
                md5='075c4b27ab2848c90ad5e87d9f922bc3',
                content=['annotation'],
                mapping=[['detext_textdet_test_gt', 'annotations/test']]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg', '.JPG'],
        rule=[r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt']),
    parser=dict(type='DetextDetAnnParser', encoding='utf-8-sig'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

delete = ['detext_textdet_train_img', 'annotations', 'detext_textdet_test_img']
config_generator = dict(type='TextDetConfigGenerator')
