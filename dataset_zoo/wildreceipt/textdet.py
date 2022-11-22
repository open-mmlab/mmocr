_base_ = ['kie.py']

data_converter = dict(
    type='TextDetDataConverter',
    parser=dict(type='WildreceiptTextDetAnnParser'),
    dumper=dict(type='JsonDumper'))

config_generator = dict(
    type='TextRecogConfigGenerator', data_root=_base_.data_root)
