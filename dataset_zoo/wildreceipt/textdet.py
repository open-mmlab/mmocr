_base_ = ['kie.py']

data_converter = dict(
    type='TextDetDataConverter',
    parser=dict(type='WildreceiptTextDetParser'),
    dumper=dict(type='JsonDumper'))
