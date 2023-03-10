_base_ = ['textdet.py']

_base_.train_preparer.packer.type = 'TextSpottingPacker'
_base_.val_preparer.packer.type = 'TextSpottingPacker'

config_generator = dict(type='TextSpottingConfigGenerator')
