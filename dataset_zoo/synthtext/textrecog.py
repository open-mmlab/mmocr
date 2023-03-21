_base_ = ['textdet.py']

_base_.train_preparer.obtainer.files.append(
    dict(
        url='https://download.openmmlab.com/mmocr/data/1.x/recog/'
        'SynthText/subset_textrecog_train.json',
        save_name='subset_textrecog_train.json',
        md5='151c4edd1cc240362046d3a6f8f4b4c6',
        split=['train'],
        content=['annotation']))
_base_.train_preparer.obtainer.files.append(
    dict(
        url='https://download.openmmlab.com/mmocr/data/1.x/recog/'
        'SynthText/alphanumeric_textrecog_train.json',
        save_name='alphanumeric_textrecog_train.json',
        md5='89b80163435794ca117a124d081d68a9',
        split=['train'],
        content=['annotation']))
_base_.train_preparer.gatherer.img_dir = 'textdet_imgs/train'
_base_.train_preparer.packer.type = 'TextRecogCropPacker'

config_generator = dict(
    type='TextRecogConfigGenerator',
    train_anns=[
        dict(ann_file='textrecog_train.json', dataset_postfix=''),
        dict(ann_file='subset_textrecog_train.json', dataset_postfix='sub'),
        dict(
            ann_file='alphanumeric_textrecog_train.json',
            dataset_postfix='an'),
    ])
