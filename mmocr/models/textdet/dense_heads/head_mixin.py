# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmocr.models.builder import HEADS
from mmocr.models.textdet.postprocess import decode
from mmocr.utils import check_argument


@HEADS.register_module()
class HeadMixin:
    """The head minxin for dbnet and pannet heads."""

    def resize_boundary(self, boundaries, scale_factor):
        """Rescale boundaries via scale_factor.

        Args:
            boundaries (list[list[float]]): The boundary list. Each boundary
                has :math:`2k+1` elements with :math:`k>=4`.
            scale_factor (ndarray): The scale factor of size :math:`(4,)`.

        Returns:
            list[list[float]]: The scaled boundaries.
        """
        assert check_argument.is_2dlist(boundaries)
        assert isinstance(scale_factor, np.ndarray)
        assert scale_factor.shape[0] == 4

        for b in boundaries:
            sz = len(b)
            check_argument.valid_boundary(b, True)
            b[:sz -
              1] = (np.array(b[:sz - 1]) *
                    (np.tile(scale_factor[:2], int(
                        (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
        return boundaries

    def get_boundary(self, score_maps, img_metas, rescale):
        """Compute text boundaries via post processing.

        Args:
            score_maps (Tensor): The text score map.
            img_metas (dict): The image meta info.
            rescale (bool): Rescale boundaries to the original image resolution
                if true, and keep the score_maps resolution if false.

        Returns:
            dict: A dict where boundary results are stored in
            ``boundary_result``.
        """

        assert check_argument.is_type_list(img_metas, dict)
        assert isinstance(rescale, bool)

        score_maps = score_maps.squeeze()
        boundaries = decode(
            decoding_type=self.decoding_type,
            preds=score_maps,
            text_repr_type=self.text_repr_type)
        if rescale:
            boundaries = self.resize_boundary(
                boundaries,
                1.0 / self.downsample_ratio / img_metas[0]['scale_factor'])
        results = dict(
            boundary_result=boundaries, filename=img_metas[0]['filename'])

        return results

    def loss(self, pred_maps, **kwargs):
        """Compute the loss for text detection.

        Args:
            pred_maps (Tensor): The input score maps of shape
                :math:`(NxCxHxW)`.

        Returns:
            dict: The dict for losses.
        """
        losses = self.loss_module(pred_maps, self.downsample_ratio, **kwargs)
        return losses
