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

IIIT5K = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix1),
    ann_file=test_ann_file1,
    test_mode=True,
    pipeline=None)

SVT = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix2),
    ann_file=test_ann_file2,
    test_mode=True,
    pipeline=None)

IC13 = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix3),
    ann_file=test_ann_file3,
    test_mode=True,
    pipeline=None)

IC15 = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix4),
    ann_file=test_ann_file4,
    test_mode=True,
    pipeline=None)

SVTP = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix5),
    ann_file=test_ann_file5,
    test_mode=True,
    pipeline=None)

CUTE80 = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix6),
    ann_file=test_ann_file6,
    test_mode=True,
    pipeline=None)

test_list = [IIIT5K, SVT, IC13, IC15, SVTP, CUTE80]

IIIT5K_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=IIIT5K)

SVT_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=SVT)

IC13_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=IC13)

IC15_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=IC15)

SVTP_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=SVTP)

CUTE80_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=CUTE80)

val_dataloader = [
    IIIT5K_val_dataloader, SVT_val_dataloader, IC13_val_dataloader,
    IC15_val_dataloader, SVTP_val_dataloader, CUTE80_val_dataloader
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
