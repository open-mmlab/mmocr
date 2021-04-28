# Modified from https://github.com/clovaai/deep-text-recognition-benchmark
#
# Licensed under the Apache License, Version 2.0 (the "License");s
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import PREPROCESSOR
from .base_preprocessor import BasePreprocessor


@PREPROCESSOR.register_module()
class TPSPreprocessor(BasePreprocessor):
    """Rectification Network of RARE, namely TPS based STN in.

    <https://arxiv.org/pdf/1603.03915.pdf>`_.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        img_size (tuple(int, int)): Size (height, width) of the input image.
        rectified_img_size (tuple(int, int))::
            Size (height, width) of the rectified image.
        num_img_channel (int): Number of channels of the input image.

    Output:
        batch_rectified_img: Rectified image with size
            [batch_size x num_img_channel x rectified_img_height
            x rectified_img_width]
    """

    def __init__(self,
                 num_fiducial=20,
                 img_size=(32, 100),
                 rectified_img_size=(32, 100),
                 num_img_channel=1):
        super().__init__()
        assert isinstance(num_fiducial, int)
        assert num_fiducial > 0
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        assert isinstance(num_img_channel, int)

        self.num_fiducial = num_fiducial
        self.img_size = img_size
        self.rectified_img_size = rectified_img_size
        self.num_img_channel = num_img_channel
        self.LocalizationNetwork = LocalizationNetwork(self.num_fiducial,
                                                       self.num_img_channel)
        self.GridGenerator = GridGenerator(self.num_fiducial,
                                           self.rectified_img_size)

    def forward(self, batch_img):
        batch_C_prime = self.LocalizationNetwork(
            batch_img)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(
            batch_C_prime, batch_img.device
        )  # batch_size x n (= rectified_img_width x rectified_img_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([
            build_P_prime.size(0), self.rectified_img_size[0],
            self.rectified_img_size[1], 2
        ])

        batch_rectified_img = F.grid_sample(
            batch_img,
            build_P_prime_reshape,
            padding_mode='border',
            align_corners=True)

        return batch_rectified_img


class LocalizationNetwork(nn.Module):
    """Localization Network of RARE, which predicts C' (K x 2) from input
    (img_width x img_height)

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        num_img_channel (int): Number of channels of the input image.
    """

    def __init__(self, num_fiducial, num_img_channel):
        super().__init__()
        self.num_fiducial = num_fiducial
        self.num_img_channel = num_img_channel
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_img_channel,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x img_height/2 x img_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x img_h/4 x img_w/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x img_h/8 x img_w/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.num_fiducial * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(num_fiducial / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(num_fiducial / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(num_fiducial / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(
            initial_bias).float().view(-1)

    def forward(self, batch_img):
        """
        Args:
            batch_img (tensor): Batch Input Image
                [batch_size x num_img_channel x img_height x img_width]
        Output:
            batch_C_prime : Predicted coordinates of fiducial points for
            input batch [batch_size x num_fiducial x 2]
        """
        batch_size = batch_img.size(0)
        features = self.conv(batch_img).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(
            self.localization_fc1(features)).view(batch_size,
                                                  self.num_fiducial, 2)
        return batch_C_prime


class GridGenerator(nn.Module):
    """Grid Generator of RARE, which produces P_prime by multipling T with P.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        rectified_img_size (tuple(int, int)):
            Size (height, width) of the rectified image.
    """

    def __init__(self, num_fiducial, rectified_img_size):
        """Generate P_hat and inv_delta_C for later."""
        super().__init__()
        self.eps = 1e-6
        self.rectified_img_height = rectified_img_size[0]
        self.rectified_img_width = rectified_img_size[1]
        self.num_fiducial = num_fiducial
        self.C = self._build_C(self.num_fiducial)  # num_fiducial x 2
        self.P = self._build_P(self.rectified_img_width,
                               self.rectified_img_height)
        # for multi-gpu, you need register buffer
        self.register_buffer(
            'inv_delta_C',
            torch.tensor(self._build_inv_delta_C(
                self.num_fiducial,
                self.C)).float())  # num_fiducial+3 x num_fiducial+3
        self.register_buffer('P_hat',
                             torch.tensor(
                                 self._build_P_hat(
                                     self.num_fiducial, self.C,
                                     self.P)).float())  # n x num_fiducial+3
        # for fine-tuning with different image width,
        # you may use below instead of self.register_buffer
        # self.inv_delta_C = torch.tensor(
        #     self._build_inv_delta_C(
        #         self.num_fiducial,
        #         self.C)).float().cuda()  # num_fiducial+3 x num_fiducial+3
        # self.P_hat = torch.tensor(
        #     self._build_P_hat(self.num_fiducial, self.C,
        #                       self.P)).float().cuda()  # n x num_fiducial+3

    def _build_C(self, num_fiducial):
        """Return coordinates of fiducial points in rectified_img; C."""
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(num_fiducial / 2))
        ctrl_pts_y_top = -1 * np.ones(int(num_fiducial / 2))
        ctrl_pts_y_bottom = np.ones(int(num_fiducial / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # num_fiducial x 2

    def _build_inv_delta_C(self, num_fiducial, C):
        """Return inv_delta_C which is needed to calculate T."""
        hat_C = np.zeros((num_fiducial, num_fiducial), dtype=float)
        for i in range(0, num_fiducial):
            for j in range(i, num_fiducial):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # num_fiducial+3 x num_fiducial+3
            [
                np.concatenate([np.ones((num_fiducial, 1)), C, hat_C],
                               axis=1),  # num_fiducial x num_fiducial+3
                np.concatenate([np.zeros(
                    (2, 3)), np.transpose(C)], axis=1),  # 2 x num_fiducial+3
                np.concatenate([np.zeros(
                    (1, 3)), np.ones((1, num_fiducial))],
                               axis=1)  # 1 x num_fiducial+3
            ],
            axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # num_fiducial+3 x num_fiducial+3

    def _build_P(self, rectified_img_width, rectified_img_height):
        rectified_img_grid_x = (
            np.arange(-rectified_img_width, rectified_img_width, 2) +
            1.0) / rectified_img_width  # self.rectified_img_width
        rectified_img_grid_y = (
            np.arange(-rectified_img_height, rectified_img_height, 2) +
            1.0) / rectified_img_height  # self.rectified_img_height
        P = np.stack(  # self.rectified_img_w x self.rectified_img_h x 2
            np.meshgrid(rectified_img_grid_x, rectified_img_grid_y),
            axis=2)
        return P.reshape([
            -1, 2
        ])  # n (= self.rectified_img_width x self.rectified_img_height) x 2

    def _build_P_hat(self, num_fiducial, C, P):
        n = P.shape[
            0]  # n (= self.rectified_img_width x self.rectified_img_height)
        P_tile = np.tile(np.expand_dims(P, axis=1),
                         (1, num_fiducial,
                          1))  # n x 2 -> n x 1 x 2 -> n x num_fiducial x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x num_fiducial x 2
        P_diff = P_tile - C_tile  # n x num_fiducial x 2
        rbf_norm = np.linalg.norm(
            P_diff, ord=2, axis=2, keepdims=False)  # n x num_fiducial
        rbf = np.multiply(np.square(rbf_norm),
                          np.log(rbf_norm + self.eps))  # n x num_fiducial
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x num_fiducial+3

    def build_P_prime(self, batch_C_prime, device='cuda'):
        """Generate Grid from batch_C_prime [batch_size x num_fiducial x 2]"""
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
            dim=1)  # batch_size x num_fiducial+3 x 2
        batch_T = torch.bmm(
            batch_inv_delta_C,
            batch_C_prime_with_zeros)  # batch_size x num_fiducial+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2
