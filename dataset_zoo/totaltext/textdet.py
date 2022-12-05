data_root = 'data/totaltext'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://universityofadelaide.box.com/shared/static/'
            '8xro7hnvb0sqw5e5rxm73tryc59j6s43.zip',
            save_name='totaltext.zip',
            md5='5b56d71a4005a333cf200ff35ce87f75',
            split=['train', 'test'],
            content=['image'],
            mapping=[['totaltext/Images/Train', 'textdet_imgs/train'],
                     ['totaltext/Images/Test', 'textdet_imgs/test']]),
        dict(
            url='https://universityofadelaide.box.com/shared/static/'
            '2vmpvjb48pcrszeegx2eznzc4izan4zf.zip',
            save_name='txt_format.zip',
            md5='53377a83420b4a0244304467512134e8',
            split=['train', 'test'],
            content=['annotation'],
            mapping=[['txt_format/Train', 'annotations/train'],
                     ['txt_format/Test', 'annotations/test']]),
    ])

data_converter = dict(
    type='TextDetDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='pair_gather',
        suffixes=['.jpg', '.JPG'],
        rule=[r'img(\d+)\.([jJ][pP][gG])', r'poly_gt_img\1.txt']),
    parser=dict(type='TotaltextTextDetAnnParser', data_root=data_root),
    dumper=dict(type='JsonDumper'),
    delete=['totaltext', 'txt_format', 'annotations'])

config_generator = dict(type='TextDetConfigGenerator', data_root=data_root)
