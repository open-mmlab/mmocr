data_root = 'data/mlt'
cache_path = 'data/cache'
train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://datasets.cvc.uab.es/rrc/'
                'ch8_training_word_images_gt_part_1.zip',
                save_name='mlt_rec_1.zip',
                md5='714d899cf5c8cf23b73bc14cfb628a3a',
                content=['image'],
                mapping=[['mlt_rec_1/*', 'textrecog_imgs/train']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/'
                'ch8_training_word_images_gt_part_2.zip',
                save_name='mlt_rec_2.zip',
                md5='d0e5bc4736626853203d24c70bbf56d1',
                content=['image'],
                mapping=[['mlt_rec_2/*', 'textrecog_imgs/train']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/'
                'ch8_training_word_images_gt_part_3.zip',
                save_name='mlt_rec_3.zip',
                md5='ebc7f2c9e73c3d174437d43b03177c5c',
                content=['image'],
                mapping=[['mlt_rec_3/*', 'textrecog_imgs/train']]),
            dict(
                url='https://rrc.cvc.uab.es/downloads/'
                'ch8_validation_word_images_gt.zip',
                save_name='mlt_rec_train_gt.zip',
                md5='e5e681b440a616f0ac8deaa669b3682d',
                content=['annotation'],
                mapping=[['mlt_rec_train_gt/', 'annotations/train']]),
        ]),
    gatherer=dict(type='MonoGatherer', ann_name='train/gt.txt'),
    parser=dict(
        type='ICDARTxtTextRecogAnnParser',
        encoding='utf-8-sig',
        format='img,lang,text'),
    packer=dict(type='TextRecogPacker'),
    dumper=dict(type='JsonDumper'),
)

val_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/'
                'ch8_validation_word_images_gt.zip',
                save_name='mlt_rec_val.zip',
                md5='954acd0325c442288fa4aff1009b6d79',
                content=['image'],
                mapping=[['mlt_rec_val/*', 'textrecog_imgs/val']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/'
                'ch8_validation_word_gt_v2.zip',
                save_name='mlt_rec_val_gt.zip',
                md5='951c9cee78a0064b133ab59369a9b232',
                content=['annotation'],
                mapping=[['mlt_rec_val_gt/', 'annotations/val']]),
        ]),
    gatherer=dict(type='MonoGatherer', ann_name='train/gt.txt'),
    parser=dict(
        type='ICDARTxtTextRecogAnnParser',
        encoding='utf-8-sig',
        format='img,lang,text'),
    packer=dict(type='TextRecogPacker'),
    dumper=dict(type='JsonDumper'),
)

config_generator = dict(
    type='TextRecogConfigGenerator',
    val_anns=[dict(ann_file='textrecog_val.json', dataset_postfix='')],
    test_anns=None)

delete = [f'mlt_rec_{i}' for i in range(1, 4)] + [
    'annotations', 'mlt_rec_val_gt', 'mlt_rec_train_gt', 'mlt_rec_val'
]
