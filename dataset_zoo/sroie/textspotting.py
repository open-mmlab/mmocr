_base_ = ['textdet.py']

_base_.train_preparer.gatherer.img_dir = 'textdet_imgs/train'
_base_.test_preparer.gatherer.img_dir = 'textdet_imgs/test'
_base_.train_preparer.packer.type = 'TextSpottingPacker'
_base_.test_preparer.packer.type = 'TextSpottingPacker'

config_generator = dict(type='TextSpottingConfigGenerator')
