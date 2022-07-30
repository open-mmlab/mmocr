# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmocr.models.textrecog.preprocessors import STN, TPStransform


class TestTPS(TestCase):

    def test_tps_transform(self):
        tps = TPStransform(output_image_size=(32, 100), num_control_points=20)
        image = torch.rand(2, 3, 32, 64)
        control_points = torch.rand(2, 20, 2)
        transformed_image = tps(image, control_points)
        self.assertEqual(transformed_image.shape, (2, 3, 32, 100))

    def test_stn(self):
        stn = STN(
            in_channels=3,
            resized_image_size=(32, 64),
            output_image_size=(32, 100),
            num_control_points=20)
        image = torch.rand(2, 3, 64, 256)
        transformed_image = stn(image)
        self.assertEqual(transformed_image.shape, (2, 3, 32, 100))
