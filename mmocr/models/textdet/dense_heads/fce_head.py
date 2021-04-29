import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmocr.models.textdet.postprocess import decode
from ..postprocess.wrapper import poly_nms
from .head_mixin import HeadMixin


@HEADS.register_module()
class FCEHead(HeadMixin, nn.Module):
    """The class for implementing FCENet head
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
    Detection

    [https://arxiv.org/abs/2104.10442]
    """

    def __init__(self,
                 in_channels,
                 scales,
                 fourier_degree=5,
                 sample_points=50,
                 reconstr_points=50,
                 decoding_type='fcenet',
                 loss=dict(type='FCELoss'),
                 score_thresh=0.3,
                 nms_thresh=0.1,
                 alpha=1.0,
                 beta=2.0,
                 train_cfg=None,
                 test_cfg=None):
        """Initialization.

        Args:
            in_channels (int): The number of input channels.
            scales (list[int]) : The scale of each layer.
            fourier_degree (int) : The maximum fourier transform degree k.
            sample_points (int) : The sampling points number of regression
                loss. If it is too small, fcenet is tend to be overfitting.
            score_thresh (float) : The threshold to filter out the final
                candidates.
            nms_thresh (float) : The threshold of nms.
            alpha (float) : The arg to calculate final score. And Score(final)
                = (Score(text region) ** alpha)
                * (Score(text center region) ** beta)
            beta (float) :The arg to calculate final score. And Score(final)
                = (Score(text region) ** alpha)
                * (Score(text center region) ** beta)
        """
        super().__init__()
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        self.k = fourier_degree
        self.n = sample_points
        self.reconstr_points = reconstr_points
        loss['fourier_degree'] = fourier_degree
        loss['sample_points'] = sample_points
        self.decoding_type = decoding_type
        self.loss_module = build_loss(loss)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.alpha = alpha
        self.beta = beta
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.out_channels_cls = 4
        self.out_channels_reg = (2 * self.k + 1) * 2

        self.out_conv_cls = nn.Conv2d(
            self.in_channels,
            self.out_channels_cls,
            kernel_size=3,
            stride=1,
            padding=1)
        self.out_conv_reg = nn.Conv2d(
            self.in_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1)

        self.init_weights()

    def init_weights(self):
        normal_init(self.out_conv_cls, mean=0, std=0.01)
        normal_init(self.out_conv_reg, mean=0, std=0.01)

    def forward(self, feats):
        cls_res, reg_res = multi_apply(self.forward_single, feats)
        level_num = len(cls_res)
        preds = [[cls_res[i], reg_res[i]] for i in range(level_num)]
        return preds

    def forward_single(self, x):
        cls_predict = self.out_conv_cls(x)
        reg_predict = self.out_conv_reg(x)
        return cls_predict, reg_predict

    def get_boundary(self, score_maps, img_metas, rescale):
        assert len(score_maps) == len(self.scales)

        boundaries = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundaries = boundaries + self._get_boundary_single(
                score_map, scale)

        # nms
        boundaries = poly_nms(boundaries, self.nms_thresh)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries)
        return results

    def _get_boundary_single(self, score_map, scale):
        assert len(score_map) == 2
        assert score_map[1].shape[1] == 4 * self.k + 2

        t = decode(
            decoding_type=self.decoding_type,
            preds=score_map,
            fourier_degree=self.k,
            reconstr_points=self.reconstr_points,
            scale=scale,
            alpha=self.alpha,
            beta=self.beta,
            text_repr_type='poly',
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh)

        return t
