# Text Recognition Testing set, including:
# Regular Datasets: IIIT5K, SVT, IC13
# Irregular Datasets: IC15, SVTP, CT80

test_root = 'data/rec'

test_img_prefix1 = 'IIIT5K/'
test_img_prefix2 = 'svt/'
test_img_prefix3 = 'icdar_2013/Challenge2_Test_Task3_Images/'
test_img_prefix4 = 'icdar_2015/ch4_test_word_images_gt'
test_img_prefix5 = 'svtp/'
test_img_prefix6 = 'ct80/'

test_ann_file1 = 'IIIT5K/test_label.json'
test_ann_file2 = 'svt/test_label.json'
test_ann_file3 = 'icdar_2013/test_label.json'
test_ann_file4 = 'icdar_2015/test_label.json'
test_ann_file5 = 'svtp/test_label.json'
test_ann_file6 = 'ct80/test_label.json'

iiit5k_rec_test = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix1),
    ann_file=test_ann_file1,
    test_mode=True,
    pipeline=None)

svt_rec_test = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix2),
    ann_file=test_ann_file2,
    test_mode=True,
    pipeline=None)

ic13_rec_test = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix3),
    ann_file=test_ann_file3,
    test_mode=True,
    pipeline=None)

ic15_rec_test = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix4),
    ann_file=test_ann_file4,
    test_mode=True,
    pipeline=None)

svtp_rec_test = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix5),
    ann_file=test_ann_file5,
    test_mode=True,
    pipeline=None)

cute80_rec_test = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix6),
    ann_file=test_ann_file6,
    test_mode=True,
    pipeline=None)

iiit5k_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=iiit5k_rec_test)

svt_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=svt_rec_test)

ic13_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ic13_rec_test)

ic15_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ic15_rec_test)

svtp_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=svtp_rec_test)

cute80_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=cute80_rec_test)

val_dataloader = [
    iiit5k_val_dataloader, svt_val_dataloader, ic13_val_dataloader,
    ic15_val_dataloader, svtp_val_dataloader, cute80_val_dataloader
]

test_dataloader = val_dataloader

val_evaluator = [[
    dict(
        type='WordMetric',
        mode=['exact', 'ignore_case', 'ignore_case_symbol'],
        prefix='IIIT5K'),
    dict(type='CharMetric', prefix='IIIT5K')
],
                 [
                     dict(
                         type='WordMetric',
                         mode=['exact', 'ignore_case', 'ignore_case_symbol'],
                         prefix='svt'),
                     dict(type='CharMetric', prefix='svt')
                 ],
                 [
                     dict(
                         type='WordMetric',
                         mode=['exact', 'ignore_case', 'ignore_case_symbol'],
                         prefix='icdar_2013'),
                     dict(type='CharMetric', prefix='icdar_2013')
                 ],
                 [
                     dict(
                         type='WordMetric',
                         mode=['exact', 'ignore_case', 'ignore_case_symbol'],
                         prefix='icdar_2015'),
                     dict(type='CharMetric', prefix='icdar_2015')
                 ],
                 [
                     dict(
                         type='WordMetric',
                         mode=['exact', 'ignore_case', 'ignore_case_symbol'],
                         prefix='svtp'),
                     dict(type='CharMetric', prefix='svtp')
                 ],
                 [
                     dict(
                         type='WordMetric',
                         mode=['exact', 'ignore_case', 'ignore_case_symbol'],
                         prefix='ct80'),
                     dict(type='CharMetric', prefix='ct80')
                 ]]
test_evaluator = val_evaluator
test_list = [
    iiit5k_rec_test, svt_rec_test, ic13_rec_test, ic15_rec_test, svtp_rec_test,
    cute80_rec_test
]
