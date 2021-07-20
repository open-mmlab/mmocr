import warnings

import mmcv
from mmdet.core import bbox2roi
from mmdet.models.builder import DETECTORS, build_roi_extractor
from mmdet.models.detectors import SingleStageDetector
from torch import nn
from torch.nn import functional as F

from mmocr.core import imshow_edge_node
from mmocr.utils import list_from_file


@DETECTORS.register_module()
class SDMGR(SingleStageDetector):
    """The implementation of the paper: Spatial Dual-Modality Graph Reasoning
    for Key Information Extraction. https://arxiv.org/abs/2103.14470.

    Args:
        visual_modality (bool): Whether use the visual modality.
        class_list (None | str): Mapping file of class index to
            class name. If None, class index will be shown in
            `show_results`, else class name.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=7),
                     featmap_strides=[1]),
                 visual_modality=False,
                 train_cfg=None,
                 test_cfg=None,
                 class_list=None,
                 init_cfg=None):
        super().__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, init_cfg=init_cfg)
        self.visual_modality = visual_modality
        if visual_modality:
            self.extractor = build_roi_extractor({
                **extractor, 'out_channels':
                self.backbone.base_channels
            })
            self.maxpool = nn.MaxPool2d(extractor['roi_layer']['output_size'])
        else:
            self.extractor = None
        self.class_list = class_list

    def forward_train(self, img, img_metas, relations, texts, gt_bboxes,
                      gt_labels):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details of the values of these keys,
                please see :class:`mmdet.datasets.pipelines.Collect`.
            relations (list[tensor]): Relations between bboxes.
            texts (list[tensor]): Texts in bboxes.
            gt_bboxes (list[tensor]): Each item is the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[tensor]): Class indices corresponding to each box.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img, gt_bboxes)
        node_preds, edge_preds = self.bbox_head.forward(relations, texts, x)
        return self.bbox_head.loss(node_preds, edge_preds, gt_labels)

    def forward_test(self,
                     img,
                     img_metas,
                     relations,
                     texts,
                     gt_bboxes,
                     rescale=False):
        x = self.extract_feat(img, gt_bboxes)
        node_preds, edge_preds = self.bbox_head.forward(relations, texts, x)
        return [
            dict(
                img_metas=img_metas,
                nodes=F.softmax(node_preds, -1),
                edges=F.softmax(edge_preds, -1))
        ]

    def extract_feat(self, img, gt_bboxes):
        if self.visual_modality:
            x = super().extract_feat(img)[-1]
            feats = self.maxpool(self.extractor([x], bbox2roi(gt_bboxes)))
            return feats.view(feats.size(0), -1)
        return None

    def show_result(self,
                    img,
                    result,
                    boxes,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    **kwargs):
        """Draw `result` on `img`.

        Args:
            img (str or tensor): The image to be displayed.
            result (dict): The results to draw on `img`.
            boxes (list): Bbox of img.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The output filename.
                Default: None.

        Returns:
            img (tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        idx_to_cls = {}
        if self.class_list is not None:
            for line in list_from_file(self.class_list):
                class_idx, class_label = line.strip().split()
                idx_to_cls[class_idx] = class_label

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        img = imshow_edge_node(
            img,
            result,
            boxes,
            idx_to_cls=idx_to_cls,
            show=show,
            win_name=win_name,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

        return img
