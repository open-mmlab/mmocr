import copy
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest

from mmocr.datasets import DATASETS, CocoDataset, CustomDataset, build_dataset


def _create_dummy_coco_json(json_name):
    image = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name.jpg',
    }

    annotation_1 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'area': 400,
        'bbox': [50, 60, 20, 20],
        'iscrowd': 0,
    }

    annotation_2 = {
        'id': 2,
        'image_id': 0,
        'category_id': 0,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
    }

    annotation_3 = {
        'id': 3,
        'image_id': 0,
        'category_id': 0,
        'area': 1600,
        'bbox': [150, 160, 40, 40],
        'iscrowd': 0,
    }

    annotation_4 = {
        'id': 4,
        'image_id': 0,
        'category_id': 0,
        'area': 10000,
        'bbox': [250, 260, 100, 100],
        'iscrowd': 0,
    }

    categories = [{
        'id': 0,
        'name': 'car',
        'supercategory': 'car',
    }]

    fake_json = {
        'images': [image],
        'annotations':
        [annotation_1, annotation_2, annotation_3, annotation_4],
        'categories': categories
    }

    mmcv.dump(fake_json, json_name)


def _create_dummy_custom_pkl(pkl_name):
    fake_pkl = [{
        'filename': 'fake_name.jpg',
        'width': 640,
        'height': 640,
        'ann': {
            'bboxes':
            np.array([[50, 60, 70, 80], [100, 120, 130, 150],
                      [150, 160, 190, 200], [250, 260, 350, 360]]),
            'labels':
            np.array([0, 0, 0, 0])
        }
    }]
    mmcv.dump(fake_pkl, pkl_name)


def _create_dummy_results():
    boxes = [
        np.array([[50, 60, 70, 80, 1.0], [100, 120, 130, 150, 0.98],
                  [150, 160, 190, 200, 0.96], [250, 260, 350, 360, 0.95]])
    ]
    return [boxes]


@pytest.mark.parametrize('config_path',
                         ['configs/_base_/det_dataset/toy_dataset.py'])
def test_dataset_init(config_path):
    data_config = mmcv.Config.fromfile(config_path)
    if 'data' not in data_config:
        return
    stage_names = ['train', 'val', 'test']
    for stage_name in stage_names:
        dataset_config = copy.deepcopy(data_config.data.get(stage_name))
        dataset = build_dataset(dataset_config)
        dataset[0]


def test_dataset_evaluation():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_dummy_coco_json(fake_json_file)

    # test single coco dataset evaluation
    coco_dataset = CocoDataset(
        ann_file=fake_json_file, classes=('car', ), pipeline=[])
    fake_results = _create_dummy_results()
    eval_results = coco_dataset.evaluate(fake_results, classwise=True)
    assert eval_results['bbox_mAP'] == 1
    assert eval_results['bbox_mAP_50'] == 1
    assert eval_results['bbox_mAP_75'] == 1

    # test concat dataset evaluation
    fake_concat_results = _create_dummy_results() + _create_dummy_results()

    # build concat dataset through two config dict
    coco_cfg = dict(
        type='CocoDataset',
        ann_file=fake_json_file,
        classes=('car', ),
        pipeline=[])
    concat_cfgs = [coco_cfg, coco_cfg]
    concat_dataset = build_dataset(concat_cfgs)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_bbox_mAP'] == 1
    assert eval_results['0_bbox_mAP_50'] == 1
    assert eval_results['0_bbox_mAP_75'] == 1
    assert eval_results['1_bbox_mAP'] == 1
    assert eval_results['1_bbox_mAP_50'] == 1
    assert eval_results['1_bbox_mAP_75'] == 1

    # build concat dataset through concatenated ann_file
    coco_cfg = dict(
        type='CocoDataset',
        ann_file=[fake_json_file, fake_json_file],
        classes=('car', ),
        pipeline=[])
    concat_dataset = build_dataset(coco_cfg)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_bbox_mAP'] == 1
    assert eval_results['0_bbox_mAP_50'] == 1
    assert eval_results['0_bbox_mAP_75'] == 1
    assert eval_results['1_bbox_mAP'] == 1
    assert eval_results['1_bbox_mAP_50'] == 1
    assert eval_results['1_bbox_mAP_75'] == 1

    # create dummy data
    fake_pkl_file = osp.join(tmp_dir.name, 'fake_data.pkl')
    _create_dummy_custom_pkl(fake_pkl_file)

    # test single custom dataset evaluation
    custom_dataset = CustomDataset(
        ann_file=fake_pkl_file, classes=('car', ), pipeline=[])
    fake_results = _create_dummy_results()
    eval_results = custom_dataset.evaluate(fake_results)
    assert eval_results['mAP'] == 1

    # test concat dataset evaluation
    fake_concat_results = _create_dummy_results() + _create_dummy_results()

    # build concat dataset through two config dict
    custom_cfg = dict(
        type='CustomDataset',
        ann_file=fake_pkl_file,
        classes=('car', ),
        pipeline=[])
    concat_cfgs = [custom_cfg, custom_cfg]
    concat_dataset = build_dataset(concat_cfgs)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_mAP'] == 1
    assert eval_results['1_mAP'] == 1

    # build concat dataset through concatenated ann_file
    concat_cfg = dict(
        type='CustomDataset',
        ann_file=[fake_pkl_file, fake_pkl_file],
        classes=('car', ),
        pipeline=[])
    concat_dataset = build_dataset(concat_cfg)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_mAP'] == 1
    assert eval_results['1_mAP'] == 1

    # build concat dataset through explict type
    concat_cfg = dict(
        type='mmdet.ConcatDataset',
        datasets=[custom_cfg, custom_cfg],
        separate_eval=False)
    concat_dataset = build_dataset(concat_cfg)
    eval_results = concat_dataset.evaluate(fake_concat_results, metric='mAP')
    assert eval_results['mAP'] == 1
    assert len(concat_dataset.datasets[0].data_infos) == \
        len(concat_dataset.datasets[1].data_infos)
    assert len(concat_dataset.datasets[0].data_infos) == 1
    tmp_dir.cleanup()


@pytest.mark.parametrize('classes, expected_length', [(['bus'], 2),
                                                      (['car'], 1),
                                                      (['bus', 'car'], 2)])
def test_allow_empty_images(classes, expected_length):
    dataset_class = DATASETS.get('CocoDataset')
    # Filter empty images
    filtered_dataset = dataset_class(
        ann_file='tests/data/coco_sample.json',
        img_prefix='tests/data',
        pipeline=[],
        classes=classes,
        filter_empty_gt=True)

    # Get all
    full_dataset = dataset_class(
        ann_file='tests/data/coco_sample.json',
        img_prefix='tests/data',
        pipeline=[],
        classes=classes,
        filter_empty_gt=False)

    assert len(filtered_dataset) == expected_length
    assert len(filtered_dataset.img_ids) == expected_length
    assert len(full_dataset) == 3
    assert len(full_dataset.img_ids) == 3
    assert filtered_dataset.CLASSES == classes
    assert full_dataset.CLASSES == classes
