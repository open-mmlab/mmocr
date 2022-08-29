cocov1_rec_train_data_root = 'data/rec/coco_text_v1'

cocov1_rec_train = dict(
    type='OCRDataset',
    data_root=cocov1_rec_train_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)
