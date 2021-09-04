# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import numpy as np

from mmocr.datasets.icdar_dataset import IcdarDataset


def _create_dummy_icdar_json(json_name):
    image_1 = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name.jpg',
    }
    image_2 = {
        'id': 1,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name1.jpg',
    }

    annotation_1 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'area': 400,
        'bbox': [50, 60, 20, 20],
        'iscrowd': 0,
        'segmentation': [[50, 60, 70, 60, 70, 80, 50, 80]]
    }

    annotation_2 = {
        'id': 2,
        'image_id': 0,
        'category_id': 0,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
        'segmentation': [[100, 120, 130, 120, 120, 150, 100, 150]]
    }

    annotation_3 = {
        'id': 3,
        'image_id': 0,
        'category_id': 0,
        'area': 1600,
        'bbox': [150, 160, 40, 40],
        'iscrowd': 1,
        'segmentation': [[150, 160, 190, 160, 190, 200, 150, 200]]
    }

    annotation_4 = {
        'id': 4,
        'image_id': 0,
        'category_id': 0,
        'area': 10000,
        'bbox': [250, 260, 100, 100],
        'iscrowd': 1,
        'segmentation': [[250, 260, 350, 260, 350, 360, 250, 360]]
    }
    annotation_5 = {
        'id': 5,
        'image_id': 1,
        'category_id': 0,
        'area': 10000,
        'bbox': [250, 260, 100, 100],
        'iscrowd': 1,
        'segmentation': [[250, 260, 350, 260, 350, 360, 250, 360]]
    }

    categories = [{
        'id': 0,
        'name': 'text',
        'supercategory': 'text',
    }]

    fake_json = {
        'images': [image_1, image_2],
        'annotations':
        [annotation_1, annotation_2, annotation_3, annotation_4, annotation_5],
        'categories':
        categories
    }

    mmcv.dump(fake_json, json_name)


def test_icdar_dataset():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_dummy_icdar_json(fake_json_file)

    # test initialization
    dataset = IcdarDataset(ann_file=fake_json_file, pipeline=[])
    assert dataset.CLASSES == ('text')
    assert dataset.img_ids == [0, 1]
    assert dataset.select_first_k == -1

    # test _parse_ann_info
    ann = dataset.get_ann_info(0)
    assert np.allclose(ann['bboxes'],
                       [[50., 60., 70., 80.], [100., 120., 130., 150.]])
    assert np.allclose(ann['labels'], [0, 0])
    assert np.allclose(ann['bboxes_ignore'],
                       [[150., 160., 190., 200.], [250., 260., 350., 360.]])
    assert np.allclose(ann['masks'],
                       [[[50, 60, 70, 60, 70, 80, 50, 80]],
                        [[100, 120, 130, 120, 120, 150, 100, 150]]])
    assert np.allclose(ann['masks_ignore'],
                       [[[150, 160, 190, 160, 190, 200, 150, 200]],
                        [[250, 260, 350, 260, 350, 360, 250, 360]]])
    assert dataset.cat_ids == [0]

    tmp_dir.cleanup()

    # test rank output
    # result = [[]]
    # out_file = tempfile.NamedTemporaryFile().name

    # with pytest.raises(AssertionError):
    #     dataset.output_ranklist(result, out_file)

    # result = [{'hmean': 1}, {'hmean': 0.5}]

    # output = dataset.output_ranklist(result, out_file)

    # assert output[0]['hmean'] == 0.5

    # test get_gt_mask
    # output = dataset.get_gt_mask()
    # assert np.allclose(output[0][0],
    #                    [[50, 60, 70, 60, 70, 80, 50, 80],
    #                     [100, 120, 130, 120, 120, 150, 100, 150]])
    # assert output[0][1] == []
    # assert np.allclose(output[1][0],
    #                    [[150, 160, 190, 160, 190, 200, 150, 200],
    #                     [250, 260, 350, 260, 350, 360, 250, 360]])
    # assert np.allclose(output[1][1],
    #                    [[250, 260, 350, 260, 350, 360, 250, 360]])

    # test evluation
    metrics = ['hmean-iou', 'hmean-ic13']
    results = [{
        'boundary_result': [[50, 60, 70, 60, 70, 80, 50, 80, 1],
                            [100, 120, 130, 120, 120, 150, 100, 150, 1]]
    }, {
        'boundary_result': []
    }]
    output = dataset.evaluate(results, metrics)

    assert output['hmean-iou:hmean'] == 1
    assert output['hmean-ic13:hmean'] == 1
