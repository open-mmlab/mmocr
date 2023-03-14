# The transcription of NAF dataset is annotated from Tessaract OCR, which is
# not accurate. The test/valid set ones were hand corrected, but the train set
# was only hand corrected a little. They aren't very good results. Better
# not to use them for recognition and text spotting.

_base_ = ['textdet.py']
_base_.train_preparer.parser.update(dict(ignore=['¿', '§'], det=False))
_base_.test_preparer.parser.update(dict(ignore=['¿', '§'], det=False))
_base_.val_preparer.parser.update(dict(ignore=['¿', '§'], det=False))
_base_.train_preparer.packer.type = 'TextRecogCropPacker'
_base_.test_preparer.packer.type = 'TextRecogCropPacker'
_base_.val_preparer.packer.type = 'TextRecogCropPacker'
_base_.train_preparer.gatherer.img_dir = 'textdet_imgs/train'
_base_.test_preparer.gatherer.img_dir = 'textdet_imgs/test'
_base_.val_preparer.gatherer.img_dir = 'textdet_imgs/val'
config_generator = dict(
    type='TextRecogConfigGenerator',
    val_anns=[dict(ann_file='textrecog_val.json', dataset_postfix='')])
