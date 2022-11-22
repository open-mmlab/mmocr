data_root = 'data/icdar2013'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'Challenge2_Training_Task12_Images.zip',
            save_name='ic13_textdet_train_img.zip',
            md5='a443b9649fda4229c9bc52751bad08fb',
            split=['train'],
            content=['image'],
            mapping=[['ic13_textdet_train_img', 'textdet_imgs/train']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'Challenge2_Training_Task1_GT.zip',
            save_name='ic13_textdet_test_img.zip',
            md5='af2e9f070c4c6a1c7bdb7b36bacf23e3',
            split=['test'],
            content=['image'],
            mapping=[['ic13_textdet_test_img', 'textdet_imgs/test']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'Challenge2_Test_Task12_Images.zip',
            save_name='ic13_textdet_train_gt.zip',
            md5='f3a425284a66cd67f455d389c972cce4',
            split=['train'],
            content=['annotation'],
            mapping=[['ic13_textdet_train_gt', 'annotations/train']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'Challenge2_Test_Task1_GT.zip',
            save_name='ic13_textdet_test_gt.zip',
            md5='3191c34cd6ac28b60f5a7db7030190fb',
            split=['test'],
            content=['annotation'],
            mapping=[['ic13_textdet_test_gt', 'annotations/test']]),
    ])

data_converter = dict(
    type='TextDetDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='pair_gather',
        suffixes=['.jpg'],
        rule=[r'(\w+)\.jpg', r'gt_\1.txt']),
    parser=dict(
        type='ICDARTxtTextDetAnnParser',
        remove_strs=[',', '"'],
        format='x1 y1 x2 y2 trans',
        separator=' ',
        mode='xyxy'),
    dumper=dict(type='JsonDumper'))

config_generator = dict(type='TextDetConfigGenerator', data_root=data_root)
