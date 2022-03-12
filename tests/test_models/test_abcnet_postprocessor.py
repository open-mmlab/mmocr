# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import mmcv
import torch

from mmocr.models.textdet.postprocess import FCOSPostprocessor


def test_abcnet_textdet_processor():
    img_metas = [{
        'img_shape': (1778, 1000, 3),
        'scale_factor': 1,
        'pad_shape': (896, 512, 3)
    }]
    cfg = mmcv.Config(
        dict(
            strides=(8, 16, 32, 64, 128),
            test_cfg=dict(
                rescale=True,
                property=['polygon', 'bboxes', 'bezier'],
                filter_and_location=True,
                reconstruct=True,
                extra_property=None,
                rescale_extra_property=False,
                nms_pre=1000,
                score_thr=0.3,
                strides=(8, 16, 32, 64, 128))))
    postprocess = FCOSPostprocessor(**cfg)
    with open('data/outputs_dict.pickle', 'rb') as f:
        outputs_dict = pickle.load(f)

    outputs_dict['cls_scores'] = outputs_dict.pop('logits_pred')
    outputs_dict['bbox_preds'] = outputs_dict.pop('reg_pred')
    outputs_dict['centerness_preds'] = outputs_dict.pop('ctrness_pred')
    outputs_dict['bezier_preds'] = outputs_dict.pop('top_feats')

    postprocess.eval()
    with torch.no_grad():
        pred_res = postprocess(outputs_dict, img_metas)
    with open('data/results.pickle', 'rb') as f:
        target_results = pickle.load(f)
    return pred_res, target_results


if __name__ == '__main__':
    test_abcnet_textdet_processor()
