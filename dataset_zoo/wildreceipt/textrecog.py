_base_ = ['textdet.py']

_base_.train_preparer.update(
    dict(
        parser=dict(type='WildreceiptTextDetAnnParser'),
        packer=dict(type='TextRecogCropPacker'),
        dumper=dict(type='JsonDumper')))

_base_.test_preparer.update(
    dict(
        parser=dict(type='WildreceiptTextDetAnnParser'),
        packer=dict(type='TextRecogCropPacker'),
        dumper=dict(type='JsonDumper')))

config_generator = dict(type='TextRecogConfigGenerator')
