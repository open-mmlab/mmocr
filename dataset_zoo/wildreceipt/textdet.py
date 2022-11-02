_base_ = ['kie.py']

data_converter = dict(
    type='TextDetDataConverter',
    parser=dict(type='WildreceiptTextDetAnnParser'),
    dumper=dict(type='JsonDumper'))
