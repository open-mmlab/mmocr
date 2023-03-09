data_root = 'data/totaltext'
cache_path = 'data/cache'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://universityofadelaide.box.com/shared/static/'
                '8xro7hnvb0sqw5e5rxm73tryc59j6s43.zip',
                save_name='totaltext.zip',
                md5='5b56d71a4005a333cf200ff35ce87f75',
                content=['image'],
                mapping=[['totaltext/Images/Train', 'textdet_imgs/train']]),
            dict(
                url='https://universityofadelaide.box.com/shared/static/'
                '2vmpvjb48pcrszeegx2eznzc4izan4zf.zip',
                save_name='txt_format.zip',
                md5='53377a83420b4a0244304467512134e8',
                content=['annotation'],
                mapping=[['txt_format/Train', 'annotations/train']]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg', '.JPG'],
        rule=[r'img(\d+)\.([jJ][pP][gG])', r'poly_gt_img\1.txt']),
    parser=dict(type='TotaltextTextDetAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://universityofadelaide.box.com/shared/static/'
                '8xro7hnvb0sqw5e5rxm73tryc59j6s43.zip',
                save_name='totaltext.zip',
                md5='5b56d71a4005a333cf200ff35ce87f75',
                content=['image'],
                mapping=[['totaltext/Images/Test', 'textdet_imgs/test']]),
            dict(
                url='https://universityofadelaide.box.com/shared/static/'
                '2vmpvjb48pcrszeegx2eznzc4izan4zf.zip',
                save_name='txt_format.zip',
                md5='53377a83420b4a0244304467512134e8',
                content=['annotation'],
                mapping=[['txt_format/Test', 'annotations/test']]),
        ]),
    gatherer=dict(
        type='PairGatherer',
        img_suffixes=['.jpg', '.JPG'],
        rule=[r'img(\d+)\.([jJ][pP][gG])', r'poly_gt_img\1.txt']),
    parser=dict(type='TotaltextTextDetAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)
delete = ['totaltext', 'txt_format', 'annotations']
config_generator = dict(type='TextDetConfigGenerator')
