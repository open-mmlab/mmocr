# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, SynthAdd, Syn90k
# Real Dataset: IC11, IC13, IC15, COCO-Test, IIIT5k
data_root = 'data/recog'

train_img_prefix1 = 'icdar_2011'
train_img_prefix2 = 'icdar_2013'
train_img_prefix3 = 'icdar_2015'
train_img_prefix4 = 'coco_text'
train_img_prefix5 = 'IIIT5K'
train_img_prefix6 = 'SynthText_Add'
train_img_prefix7 = 'SynthText'
train_img_prefix8 = 'Syn90k'

train_ann_file1 = 'icdar_2011/train_label.json',
train_ann_file2 = 'icdar_2013/train_label.json',
train_ann_file3 = 'icdar_2015/train_label.json',
train_ann_file4 = 'coco_text/train_label.json',
train_ann_file5 = 'IIIT5K/train_label.json',
train_ann_file6 = 'SynthText_Add/label.json',
train_ann_file7 = 'SynthText/shuffle_labels.json',
train_ann_file8 = 'Syn90k/shuffle_labels.json'

train1 = dict(
    type='OCRDataset',
    data_root=data_root,
    data_prefix=dict(img_path=train_img_prefix1),
    ann_file=train_ann_file1,
    test_mode=False,
    pipeline=None)

train2 = {key: value for key, value in train1.items()}
train2['data_prefix'] = dict(img_path=train_img_prefix2)
train2['ann_file'] = train_ann_file2

train3 = {key: value for key, value in train1.items()}
train3['img_prefix'] = dict(img_path=train_img_prefix3)
train3['ann_file'] = train_ann_file3

train4 = {key: value for key, value in train1.items()}
train4['img_prefix'] = dict(img_path=train_img_prefix4)
train4['ann_file'] = train_ann_file4

train5 = {key: value for key, value in train1.items()}
train5['img_prefix'] = dict(img_path=train_img_prefix5)
train5['ann_file'] = train_ann_file5

train6 = {key: value for key, value in train1.items()}
train6['img_prefix'] = dict(img_path=train_img_prefix6)
train6['ann_file'] = train_ann_file6

train7 = {key: value for key, value in train1.items()}
train7['img_prefix'] = dict(img_path=train_img_prefix7)
train7['ann_file'] = train_ann_file7

train8 = {key: value for key, value in train1.items()}
train8['img_prefix'] = dict(img_path=train_img_prefix8)
train8['ann_file'] = train_ann_file8

train_list = [train1, train2, train3, train4, train5, train6, train7, train8]
