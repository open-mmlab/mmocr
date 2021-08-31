# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import pytest

from mmocr.datasets import DATASETS


# Adapted from mmdetection
@patch('mmocr.datasets.CocoDataset.load_annotations', MagicMock())
@patch('mmocr.datasets.CocoDataset._filter_imgs', MagicMock)
@patch('mmocr.datasets.CustomDataset._filter_imgs', MagicMock)
@pytest.mark.parametrize('dataset', ['CocoDataset'])
def test_custom_classes_override_default(dataset):
    dataset_class = DATASETS.get(dataset)
    if dataset in ['CocoDataset']:
        dataset_class.coco = MagicMock()
        dataset_class.cat_ids = MagicMock()

    original_classes = dataset_class.CLASSES

    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=('bus', 'car'),
        test_mode=True,
        img_prefix='')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ('bus', 'car')

    # Test setting classes as a list
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=['bus', 'car'],
        test_mode=True,
        img_prefix='')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']

    # Test overriding not a subset
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=['foo'],
        test_mode=True,
        img_prefix='')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['foo']

    # Test default behavior
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=None,
        test_mode=True,
        img_prefix='')

    assert custom_dataset.CLASSES == original_classes
    print(custom_dataset)

    # Test sending file path
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('bus\ncar\n')
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=tmp_file.name,
        test_mode=True,
        img_prefix='')
    tmp_file.close()

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
