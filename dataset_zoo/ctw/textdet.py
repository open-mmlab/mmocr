data_root = 'data/ctw'
cache_path = 'data/cache'

data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://universityofadelaide.box.com/shared/static/'
            'py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip',
            save_name='train_images.zip',
            md5='f1453464b764343040644464d5c0c4fa',
            split=['train'],
            content=['image'],
            mapping=[['train_images/train_images', 'textdet_imgs/train']]),
        dict(
            url='https://universityofadelaide.box.com/shared/static/'
            'jikuazluzyj4lq6umzei7m2ppmt3afyw.zip',
            save_name='train_labels.zip',
            md5='d9ba721b25be95c2d78aeb54f812a5b1',
            split=['train'],
            content=['annotation'],
            mapping=[[
                'train_labels/ctw1500_train_labels/', 'annotations/train'
            ]]),
        dict(
            url='https://universityofadelaide.box.com/shared/static/'
            't4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip',
            save_name='test_images.zip',
            md5='79103fd77dfdd2c70ae6feb3a2fb4530',
            split=['test'],
            content=['image'],
            mapping=[['test_images/test_images', 'textdet_imgs/test']]),
        dict(
            url='https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/'
            'download',
            save_name='test_labels.zip',
            md5='7f650933a30cf1bcdbb7874e4962a52b',
            split=['test'],
            content=['annotation'],
            mapping=[['test_labels', 'annotations/test']]),
    ])

data_converter = dict(
    type='TextDetDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='pair_gather',
        suffixes=['.jpg'],
        rule=[r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt']),
    parser=dict(type='NAFAnnParser', data_root=data_root, det=True),
    delete=['temp_images', 'data_split.json', 'annotations', 'naf_anno'],
    dumper=dict(type='JsonDumper'),
    nproc=1)

# config_generator = dict(
#     type='TextDetConfigGenerator',
#     data_root=data_root,
#     val_anns=[dict(ann_file='textdet_val.json', dataset_postfix='')])
