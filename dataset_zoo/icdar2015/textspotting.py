_base_ = ['textdet.py']

_base_.train_preparer.gatherer.img_dir = 'textdet_imgs/train'
_base_.train_preparer.packer.type = 'TextSpottingPacker'

_base_.test_preparer.gatherer.img_dir = 'textdet_imgs/test'
_base_.test_preparer.packer.type = 'TextSpottingPacker'
_base_.test_preparer.obtainer.files = [
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
    dict(
        url='https://download.openmmlab.com/mmocr/data/1.x/'
        'textspotting/icdar2015/lexicons.tar.gz',
        save_name='icdar2015_lexicons.tar.gz',
        md5='d1e7b5e023b1d40fbdb120bd5b0878e2',
        content=['annotation'],
        mapping=[['icdar2015_lexicons/lexicons', 'lexicons']]),
]

config_generator = dict(type='TextSpottingConfigGenerator')
delete = [
    'annotations', 'ic15_textdet_test_img', 'ic15_textdet_train_img',
    'icdar2015_lexicons'
]
