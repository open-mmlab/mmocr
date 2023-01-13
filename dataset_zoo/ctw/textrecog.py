_base_ = ['textdet.py']
data_root = 'data/ctw'

data_converter = dict(type='TextRecogCropConverter')

config_generator = dict(
    type='TextRecogConfigGenerator',
    data_root=data_root,
    val_anns=[dict(ann_file='textrecog_val.json', dataset_postfix='')])
