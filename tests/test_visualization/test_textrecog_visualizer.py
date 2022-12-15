# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest

import cv2
import numpy as np
from mmengine.structures import LabelData

from mmocr.structures import TextRecogDataSample
from mmocr.visualization import TextRecogLocalVisualizer


class TestTextDetLocalVisualizer(unittest.TestCase):

    def test_add_datasample(self):
        h, w = 64, 128
        image = np.random.randint(0, 256, size=(h, w, 3)).astype('uint8')

        # test gt_text
        data_sample = TextRecogDataSample()
        img_meta = dict(img_shape=(12, 10, 3))
        gt_text = LabelData(metainfo=img_meta)
        gt_text.item = 'mmocr'
        data_sample.gt_text = gt_text

        recog_local_visualizer = TextRecogLocalVisualizer()
        recog_local_visualizer.add_datasample('image', image, data_sample)

        # test gt_text and pred_text
        pred_text = LabelData(metainfo=img_meta)
        pred_text.item = 'MMOCR'
        data_sample.pred_text = pred_text

        with tempfile.TemporaryDirectory() as tmp_dir:
            # test out
            out_file = osp.join(tmp_dir, 'out_file.jpg')

            # draw_gt = True + gt_sample
            recog_local_visualizer.add_datasample(
                'image',
                image,
                data_sample,
                out_file=out_file,
                draw_gt=True,
                draw_pred=False)
            self._assert_image_and_shape(out_file, (h * 2, w, 3))

            # draw_gt = True
            recog_local_visualizer.add_datasample(
                'image',
                image,
                data_sample,
                out_file=out_file,
                draw_gt=True,
                draw_pred=True)
            self._assert_image_and_shape(out_file, (h * 3, w, 3))

            # draw_gt = False
            recog_local_visualizer.add_datasample(
                'image', image, data_sample, draw_gt=False, out_file=out_file)
            self._assert_image_and_shape(out_file, (h * 2, w, 3))

            # gray image
            image = np.random.randint(0, 256, size=(h, w)).astype('uint8')
            recog_local_visualizer.add_datasample(
                'image', image, data_sample, draw_gt=False, out_file=out_file)
            self._assert_image_and_shape(out_file, (h * 2, w, 3))

    def _assert_image_and_shape(self, out_file, out_shape):
        self.assertTrue(osp.exists(out_file))
        drawn_img = cv2.imread(out_file)
        self.assertTrue(drawn_img.shape == out_shape)
