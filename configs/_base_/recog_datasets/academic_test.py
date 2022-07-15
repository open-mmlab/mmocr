# Text Recognition Testing set, including:
# Regular Datasets: IIIT5K, SVT, IC13
# Irregular Datasets: IC15, SVTP, CT80

test_root = 'data/recog'

test_img_prefix1 = 'IIIT5K/'
test_img_prefix2 = 'svt/'
test_img_prefix3 = 'icdar_2013/'
test_img_prefix4 = 'icdar_2015/'
test_img_prefix5 = 'svtp/'
test_img_prefix6 = 'ct80/'

test_ann_file1 = 'IIIT5K/test_label.josn'
test_ann_file2 = 'svt/test_label.josn'
test_ann_file3 = 'icdar_2013/test_label_1015.josn'
test_ann_file4 = 'icdar_2015/test_label.josn'
test_ann_file5 = 'svtp/test_label.josn'
test_ann_file6 = 'ct80/test_label.josn'

test1 = dict(
    type='OCRDataset',
    data_root=test_root,
    data_prefix=dict(img_path=test_img_prefix1),
    ann_file=test_ann_file1,
    test_mode=True,
    pipeline=None)

test2 = {key: value for key, value in test1.items()}
test2['data_prefix'] = dict(img_path=test_img_prefix2)
test2['ann_file'] = test_ann_file2

test3 = {key: value for key, value in test1.items()}
test3['data_prefix'] = dict(img_path=test_img_prefix3)
test3['ann_file'] = test_ann_file3

test4 = {key: value for key, value in test1.items()}
test4['data_prefix'] = dict(img_path=test_img_prefix4)
test4['ann_file'] = test_ann_file4

test5 = {key: value for key, value in test1.items()}
test5['data_prefix'] = dict(img_path=test_img_prefix5)
test5['ann_file'] = test_ann_file5

test6 = {key: value for key, value in test1.items()}
test6['data_prefix'] = dict(img_path=test_img_prefix6)
test6['ann_file'] = test_ann_file6

test_list = [test1, test2, test3, test4, test5, test6]
