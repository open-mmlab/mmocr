sroie_textrecog_data_root = 'data/sroie'

sroie_textrecog_train = dict(
    type='OCRDataset',
    data_root=sroie_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

sroie_textrecog_test = dict(
    type='OCRDataset',
    data_root=sroie_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
