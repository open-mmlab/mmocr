_base_ = ['textdet.py']

_base_.train_preparer.packer.type = 'TextSpottingPacker'

config_generator = dict(type='TextSpottingConfigGenerator')
