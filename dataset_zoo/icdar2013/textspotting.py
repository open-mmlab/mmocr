_base_ = ['textdet.py']

_base_.train_preparer.gatherer.img_dir = 'textdet_imgs/train'
_base_.train_preparer.packer.type = 'TextSpottingPacker'

_base_.test_preparer.packer.type = 'TextSpottingPacker'
_base_.test_preparer.gatherer.img_dir = 'textdet_imgs/test'
_base_.test_preparer.obtainer.files = [
    dict(
        url='https://rrc.cvc.uab.es/downloads/'
        'Challenge2_Test_Task12_Images.zip',
        save_name='ic13_textdet_test_img.zip',
        md5='af2e9f070c4c6a1c7bdb7b36bacf23e3',
        content=['image'],
        mapping=[['ic13_textdet_test_img', 'textdet_imgs/test']]),
    dict(
        url='https://download.openmmlab.com/mmocr/data/1.x/'
        'textspotting/icdar2013/ic13_textspotting_test_gt.zip',
        save_name='ic13_textspotting_test_gt.zip',
        md5='d0d95e800504795d153f4f21d4d8ce07',
        content=['annotation'],
        mapping=[['ic13_textspotting_test_gt', 'annotations/test']]),
    dict(
        url='https://download.openmmlab.com/mmocr/data/1.x/'
        'textspotting/icdar2013/lexicons.zip',
        save_name='icdar2013_lexicons.zip',
        md5='4ba017ab0637dae7a66fa471ac0a3253',
        content=['annotation'],
        mapping=[['icdar2013_lexicons/lexicons', 'lexicons']]),
]
_base_.test_preparer.parser = dict(
    type='ICDARTxtTextDetAnnParser', encoding='utf-8-sig')

delete = [
    'annotations', 'ic13_textdet_train_img', 'ic13_textdet_train_gt',
    'ic13_textdet_test_img', 'ic13_textdet_test_gt', 'icdar2013_lexicons'
]
config_generator = dict(type='TextSpottingConfigGenerator')
