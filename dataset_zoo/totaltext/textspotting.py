_base_ = ['textdet.py']

# _base_.train_preparer.gatherer.img_dir = 'textdet_imgs/train'
# _base_.train_preparer.packer.type = 'TextSpottingPacker'
_base_.train_preparer = None

_base_.test_preparer.gatherer.img_dir = 'textdet_imgs/test'
_base_.test_preparer.packer.type = 'TextSpottingPacker'
_base_.test_preparer.obtainer.files = [
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
    dict(
        url='https://download.openmmlab.com/mmocr/data/1.x/'
        'textspotting/totaltext/lexicons.tar.gz',
        save_name='totaltext_lexicons.tar.gz',
        md5='59e3dd01ee83355043ac69437e2e2ff2',
        content=['annotation'],
        mapping=[['totaltext_lexicons/lexicons', 'lexicons']]),
]

delete = ['totaltext', 'txt_format', 'annotations', 'totaltext_lexicons']

config_generator = dict(type='TextSpottingConfigGenerator')
