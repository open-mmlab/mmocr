import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss

from mmocr.models.textdet.postprocess import decode
from ..postprocess.wrapper import poly_nms
from .head_mixin import HeadMixin


@HEADS.register_module()
class FCEHead(HeadMixin, BaseModule):
    """The class for implementing FCENet head.
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
    Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        in_channels (int): The number of input channels.
        scales (list[int]) : The scale of each layer.
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, FCEnet tends to be overfitting.
        score_thr (float) : The threshold to filter out the final
            candidates.
        nms_thr (float) : The threshold of nms.
        alpha (float) : The parameter to calculate final scores. Score_{final}
            = (Score_{text region} ^ alpha)
            * (Score{text center region} ^ beta)
        beta (float) :The parameter to calculate final scores.
    """

    def __init__(
        self,
        in_channels,
        scales,
        fourier_degree=5,
        num_sample=50,
        num_reconstr_points=50,
        decoding_type='fcenet',
        loss=dict(type='FCELoss'),
        score_thr=0.3,
        nms_thr=0.1,
        alpha=1.0,
        beta=1.0,
        text_repr_type='poly',
        train_cfg=None,
        test_cfg=None,
        init_cfg=dict(
            type='Normal',
            mean=0,
            std=0.01,
            override=[dict(name='out_conv_cls'),
                      dict(name='out_conv_reg')])):

        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        self.fourier_degree = fourier_degree
        self.sample_num = num_sample
        self.num_reconstr_points = num_reconstr_points
        loss['fourier_degree'] = fourier_degree
        loss['num_sample'] = num_sample
        self.decoding_type = decoding_type
        self.loss_module = build_loss(loss)
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.alpha = alpha
        self.beta = beta
        self.text_repr_type = text_repr_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.out_channels_cls = 4
        self.out_channels_reg = (2 * self.fourier_degree + 1) * 2

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

    '''
    def init_weights(self):
        normal_init(self.out_conv_cls, mean=0, std=0.01)
        normal_init(self.out_conv_reg, mean=0, std=0.01)
    '''

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
        boundaries = poly_nms(boundaries, self.nms_thr)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries)
        return results

    def _get_boundary_single(self, score_map, scale):
        assert len(score_map) == 2
        assert score_map[1].shape[1] == 4 * self.fourier_degree + 2

        return decode(
            decoding_type=self.decoding_type,
            preds=score_map,
            fourier_degree=self.fourier_degree,
            num_reconstr_points=self.num_reconstr_points,
            scale=scale,
            alpha=self.alpha,
            beta=self.beta,
            text_repr_type=self.text_repr_type,
            score_thr=self.score_thr,
            nms_thr=self.nms_thr)
