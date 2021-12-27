# Copyright (c) OpenMMLab. All rights reserved.
import importlib

import numpy as np
import torch.nn as nn
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.utils import imdenormalize_torch, resize_boundary_to_score_map

from mmocr.apis.inference import model_inference
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.utils import (boundary_to_bbox, merge_text_spotter_result,
                         revert_sync_batchnorm)

BUILD_MODEL = {
    'textdet': ('mmocr.models', 'build_detector'),
    'textrecog': ('mmocr.models', 'build_detector')
}


class TwoStepTextSpotter(nn.Module):
    """two step text spotter for test.

    Args:
        text_det_config (mmcv.Config): text detection config.
        text_det_ckpt_path (str): text detection model ckpt path
        text_recog_config (mmcv.Config): text recognition config.
        text_recog_ckpt_path (str): text recognition  model ckpt path
    Returns:
        list[dict]: end-to-end text recognition results of
        text detection and text recognition
        [
            {
                "filename": "img_xxx.jpg"
                "result":
                    [{
                        "box": [159, 82, 488, 428 ...],
                        "box_score":"0.620622",
                        "text":"horse123",
                        "text_score": "0.88"}
                    ],
            }
        ]
    """

    def __init__(self,
                 config_list,
                 checkpoint_path_list,
                 task_list=['textdet', 'textrecog']):
        super().__init__()
        # length of configs should be consist with checkpoints
        assert len(config_list) == len(checkpoint_path_list)

        # build models and load checkpoint
        self.models = nn.ModuleList()
        for config, ckpt_path, task in zip(config_list, checkpoint_path_list,
                                           task_list):
            config.model.train_cfg = None
            module = importlib.import_module(BUILD_MODEL[task][0])
            build_model = getattr(module, BUILD_MODEL[task][1])
            model_kwargs = dict()
            if config.get('test_cfg', None):
                model_kwargs['test_cfg'] = config.get('test_cfg')
            model = build_model(config.model, **model_kwargs)
            load_checkpoint(model, ckpt_path, map_location='cpu')
            model.cfg = config
            model = revert_sync_batchnorm(model)
            fp16_cfg = model.cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)
            self.models.append(model)
        self.recog_bs = 1

    def forward(self, imgs, img_metas, return_loss=False, **kwargs):
        assert len(imgs) == 1, 'does not support test-time augmentation'
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        imgs, img_metas = imgs[0], img_metas[0]
        textdet_model, textrecog_model = self.models[0], self.models[1]
        output = textdet_model(imgs, img_metas, return_loss, **kwargs)

        imgs = imdenormalize_torch(imgs, img_metas['img_norm_cfg']['mean'],
                                   img_metas['img_norm_cfg']['std'])

        boundary_result = [res['boundary_result'] for res in output]
        for info, boundary in zip(boundary_result, img_metas):
            boundary = resize_boundary_to_score_map(boundary[:],
                                                    info['scale_factor'])
        bboxes = boundary_to_bbox(boundary_result)
        bboxes_imgs = list()
        for bbox, img in zip(bboxes, imgs):
            bboxes_imgs.extend(
                crop_img(img.squeeze(), bbox).cpu().numpy().astype(np.uint8))
        recog_result = list()
        arr_chunks = [
            bboxes_imgs[i:i + self.recog_bs]
            for i in range(0, len(bboxes_imgs), self.recog_bs)
        ]
        for chunk in arr_chunks:
            recog_result.extend(
                model_inference(textrecog_model, chunk, batch_mode=True))
        num_proposals_per_img = tuple(len(p) for p in boundary_result)
        e2e_results = list()
        recog_result = recog_result.split(num_proposals_per_img, 0)
        for i in len(imgs):
            e2e_results.append(
                merge_text_spotter_result(boundary_result[i], recog_result[i]))
        return e2e_results
