cocotextv1_textrecog_data_root = 'data/rec/coco_text_v1'

cocotextv1_textrecog_train = dict(
    type='OCRDataset',
    data_root=cocotextv1_textrecog_data_root,
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)
