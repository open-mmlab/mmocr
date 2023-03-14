_base_ = ['textdet.py']

_base_.train_preparer.packer.type = 'TextRecogCropPacker'
_base_.val_preparer.packer.type = 'TextRecogCropPacker'

config_generator = dict(type='TextRecogConfigGenerator')
