_base_ = ['textdet.py']
data_root = 'data/ctw'

data_converter = dict(type='TextSpottingDataConverter')

config_generator = dict(
    type='TextSpottingConfigGenerator',
    data_root=data_root,
    val_anns=[dict(ann_file='textspotting_val.json', dataset_postfix='')])
