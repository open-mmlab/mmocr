_base_ = ['textdet.py']

_base_.train_preparer.gatherer.img_dir = 'textdet_imgs/train'
_base_.test_preparer.gatherer.img_dir = 'textdet_imgs/test'

_base_.train_preparer.packer.type = 'TextSpottingPacker'
_base_.test_preparer.packer.type = 'TextSpottingPacker'

_base_.test_preparer.obtainer.files.append(
    dict(
        url='https://download.openmmlab.com/mmocr/data/1.x/textspotting/'
        'ctw1500/lexicons.zip',
        save_name='ctw1500_lexicons.zip',
        md5='168150ca45da161917bf35a20e45b8d6',
        content=['lexicons'],
        mapping=[['ctw1500_lexicons/lexicons', 'lexicons']]))

_base_.delete.append('ctw1500_lexicons')
config_generator = dict(type='TextSpottingConfigGenerator')
