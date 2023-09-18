data_root = 'data/mlt'
cache_path = 'data/cache'
# yapf: disable
train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://datasets.cvc.uab.es/rrc/ch8_training_images_1.zip',  # noqa: E501
                save_name='mlt_1.zip',
                md5='7b26e10d949c00fb4411f40b4f1fce6e',
                content=['image'],
                mapping=[['mlt_1/*', 'textdet_imgs/train']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/ch8_training_images_2.zip',  # noqa: E501
                save_name='mlt_2.zip',
                md5='e992fb5a7621dd6329081a73e52a28e1',
                content=['image'],
                mapping=[['mlt_2/*', 'textdet_imgs/train']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/ch8_training_images_3.zip',  # noqa: E501
                save_name='mlt_3.zip',
                md5='044ea5fb1dcec8bbb874391c517b55ff',
                content=['image'],
                mapping=[['mlt_3/*', 'textdet_imgs/train']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/ch8_training_images_4.zip',  # noqa: E501
                save_name='mlt_4.zip',
                md5='344a657c1cc7cbb150547f1c76b5cc8e',
                content=['image'],
                mapping=[['mlt_4/*', 'textdet_imgs/train']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/ch8_training_images_5.zip',  # noqa: E501
                save_name='mlt_5.zip',
                md5='5c7ac0158e7189c0a634eaf7bdededc5',
                content=['image'],
                mapping=[['mlt_5/*', 'textdet_imgs/train']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/ch8_training_images_6.zip',  # noqa: E501
                save_name='mlt_6.zip',
                md5='3b479255a96d255680f51005b5232bac',
                content=['image'],
                mapping=[['mlt_6/*', 'textdet_imgs/train']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/ch8_training_images_7.zip',  # noqa
                save_name='mlt_7.zip',
                md5='faa033fb9d2922d747bad9b0692c992e',
                content=['image'],
                mapping=[['mlt_7/*', 'textdet_imgs/train']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/ch8_training_images_8.zip',  # noqa
                save_name='mlt_8.zip',
                md5='db8afa59ae520757151f6ce5acd489ef',
                content=['image'],
                mapping=[['mlt_8/*', 'textdet_imgs/train']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/'
                'ch8_training_localization_transcription_gt_v2.zip',
                save_name='mlt_train_gt.zip',
                md5='2c9c3de30b5615f6846738bbd336c988',
                content=['annotation'],
                mapping=[['mlt_train_gt/', 'annotations/train']]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg', '.JPG'],
        rule=[r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt']),
    parser=dict(
        type='ICDARTxtTextDetAnnParser',
        encoding='utf-8-sig',
        format='x1,y1,x2,y2,x3,y3,x4,y4,lang,trans'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)  # noqa

val_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch8_validation_images.zip',  # noqa
                save_name='mlt_val_img.zip',
                md5='3cfc7b440ab81b89a981d707786dbe83',
                content=['image'],
                mapping=[['mlt_val_img', 'textdet_imgs/val']]),
            dict(
                url='https://datasets.cvc.uab.es/rrc/'
                'ch8_validation_localization_transcription_gt_v2.zip',
                save_name='mlt_val_gt.zip',
                md5='ecae7d433e6f103bb31e00d37254009c',
                content=['annotation'],
                mapping=[['mlt_val_gt/', 'annotations/val']]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg', '.JPG'],
        rule=[r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt']),
    parser=dict(
        type='ICDARTxtTextDetAnnParser',
        encoding='utf-8-sig',
        format='x1,y1,x2,y2,x3,y3,x4,y4,lang,trans'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

config_generator = dict(
    type='TextDetConfigGenerator',
    val_anns=[dict(ann_file='textdet_val.json', dataset_postfix='')],
    test_anns=None)

delete = [f'mlt{i}' for i in range(1, 9)
          ] + ['annotations', 'mlt_val_gt', 'mlt_train_gt']
