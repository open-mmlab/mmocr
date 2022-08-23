ic13_rec_data_root = 'data/rec/icdar_2013/'

ic13_rec_train = dict(
    type='OCRDataset',
    data_root=ic13_rec_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

ic13_rec_test = dict(
    type='OCRDataset',
    data_root=ic13_rec_data_root,
    data_prefix=dict(img_path='Challenge2_Test_Task3_Images/'),
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
