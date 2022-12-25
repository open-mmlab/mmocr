_base_ = ['textdet.py']

data_converter = dict(type='TextRecogCropConverter')

config_generator = dict(type='TextRecogConfigGenerator')