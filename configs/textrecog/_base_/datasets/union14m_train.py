union14m_data_root = 'data/Union14M-L/'

union14m_challenging = dict(
    type='OCRDataset',
    data_root=union14m_data_root,
    ann_file='train_annos/mmocr1.0/train_challenging.json',
    test_mode=True,
    pipeline=None)

union14m_hard = dict(
    type='OCRDataset',
    data_root=union14m_data_root,
    ann_file='train_annos/mmocr1.0/train_hard.json',
    pipeline=None)

union14m_medium = dict(
    type='OCRDataset',
    data_root=union14m_data_root,
    ann_file='train_annos/mmocr1.0/train_medium.json',
    pipeline=None)

union14m_normal = dict(
    type='OCRDataset',
    data_root=union14m_data_root,
    ann_file='train_annos/mmocr1.0/train_normal.json',
    pipeline=None)

union14m_easy = dict(
    type='OCRDataset',
    data_root=union14m_data_root,
    ann_file='train_annos/mmocr1.0/train_easy.json',
    pipeline=None)

union14m_val = dict(
    type='OCRDataset',
    data_root=union14m_data_root,
    ann_file='train_annos/mmocr1.0/val_annos.json',
    pipeline=None)
