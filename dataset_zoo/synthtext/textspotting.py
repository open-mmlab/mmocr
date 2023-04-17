_base_ = ['textdet.py']

_base_.train_preparer.packer.type = 'TextSpottingPacker'
_base_.train_preparer.gatherer.img_dir = 'textdet_imgs/train'

config_generator = dict(type='TextSpottingConfigGenerator')
