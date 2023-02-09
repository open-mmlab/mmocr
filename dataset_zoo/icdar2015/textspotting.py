_base_ = ['textdet.py']
_base_.prepare_train_data.packer.type = 'TextSpottingPacker'
_base_.prepare_test_data.packer.type = 'TextSpottingPacker'

config_generator = dict(type='TextSpottingConfigGenerator')
