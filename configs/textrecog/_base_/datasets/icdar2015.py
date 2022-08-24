ic15_rec_data_root = 'data/rec/icdar_2015/'

ic15_rec_train = dict(
    type='OCRDataset',
    data_root=ic15_rec_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

ic15_rec_test = dict(
    type='OCRDataset',
    data_root=ic15_rec_data_root,
    data_prefix=dict(img_path='ch4_test_word_images_gt/'),
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
