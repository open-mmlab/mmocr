import json
import os.path as osp
import tempfile

import numpy as np

from mmocr.datasets.text_det_dataset import TextDetDataset


def _create_dummy_ann_file(ann_file):
    ann_info1 = {
        'file_name':
        'sample1.jpg',
        'height':
        640,
        'width':
        640,
        'annotations': [{
            'iscrowd': 0,
            'category_id': 1,
            'bbox': [50, 70, 80, 100],
            'segmentation': [[50, 70, 80, 70, 80, 100, 50, 100]]
        }, {
            'iscrowd':
            1,
            'category_id':
            1,
            'bbox': [120, 140, 200, 200],
            'segmentation': [[120, 140, 200, 140, 200, 200, 120, 200]]
        }]
    }

    with open(ann_file, 'w') as fw:
        fw.write(json.dumps(ann_info1) + '\n')


def _create_dummy_loader():
    loader = dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations']))
    return loader


def test_detect_dataset():
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    ann_file = osp.join(tmp_dir.name, 'fake_data.txt')
    _create_dummy_ann_file(ann_file)

    # test initialization
    loader = _create_dummy_loader()
    dataset = TextDetDataset(ann_file, loader, pipeline=[])

    # test _parse_ann_info
    img_ann_info = dataset.data_infos[0]
    ann = dataset._parse_anno_info(img_ann_info['annotations'])
    print(ann['bboxes'])
    assert np.allclose(ann['bboxes'], [[50., 70., 80., 100.]])
    assert np.allclose(ann['labels'], [1])
    assert np.allclose(ann['bboxes_ignore'], [[120, 140, 200, 200]])
    assert np.allclose(ann['masks'], [[[50, 70, 80, 70, 80, 100, 50, 100]]])
    assert np.allclose(ann['masks_ignore'],
                       [[[120, 140, 200, 140, 200, 200, 120, 200]]])

    tmp_dir.cleanup()

    # test prepare_train_img
    pipeline_results = dataset.prepare_train_img(0)
    assert np.allclose(pipeline_results['bbox_fields'], [])
    assert np.allclose(pipeline_results['mask_fields'], [])
    assert np.allclose(pipeline_results['seg_fields'], [])
    expect_img_info = {'filename': 'sample1.jpg', 'height': 640, 'width': 640}
    assert pipeline_results['img_info'] == expect_img_info

    # test evluation
    metrics = 'hmean-iou'
    results = [{'boundary_result': [[50, 70, 80, 70, 80, 100, 50, 100, 1]]}]
    eval_res = dataset.evaluate(results, metrics)

    assert eval_res['hmean-iou:hmean'] == 1
