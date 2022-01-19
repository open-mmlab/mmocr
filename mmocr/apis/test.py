# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.image import tensor2imgs
from mmdet.core import encode_mask_results

from .utils import tensor2grayimgs


def retrieve_img_tensor_and_meta(data):
    """Retrieval img_tensor, img_metas and img_norm_cfg.

    Args:
        data (dict): One batch data from data_loader.

    Returns:
        tuple: Returns (img_tensor, img_metas, img_norm_cfg).

            - | img_tensor (Tensor): Input image tensor with shape
                :math:`(N, C, H, W)`.
            - | img_metas (list[dict]): The metadata of images.
            - | img_norm_cfg (dict): Config for image normalization.
    """

    img_tensor = data['img'][0].data[0]
    img_metas = data['img_metas'][0].data[0]

    # For textdet pipeline using "ImageToTensor" but not "DefaultFormatBundle"
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    must_keys = ['img_norm_cfg', 'ori_filename', 'img_shape', 'ori_shape']
    for key in must_keys:
        if key not in img_metas[0]:
            raise KeyError(
                f'Please add {key} to the "meta_keys" in the pipeline')

    img_norm_cfg = img_metas[0]['img_norm_cfg']

    return img_tensor, img_metas, img_norm_cfg


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    is_kie=False,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if is_kie:
                img_tensor = data['img'].data[0]
                if img_tensor.shape[0] != 1:
                    raise KeyError('Visualizing KIE outputs in batches is'
                                   'currently not supported.')
                gt_bboxes = data['gt_bboxes'].data[0]
                img_metas = data['img_metas'].data[0]
                must_keys = ['img_norm_cfg', 'ori_filename', 'img_shape']
                for key in must_keys:
                    if key not in img_metas[0]:
                        raise KeyError(
                            f'Please add {key} to the "meta_keys" in config.')
                # for no visual model
                if np.prod(img_tensor.shape) == 0:
                    imgs = []
                    for img_meta in img_metas:
                        try:
                            img = mmcv.imread(img_meta['filename'])
                        except Exception as e:
                            print(f'Load image with error: {e}, '
                                  'use empty image instead.')
                            img = np.ones(
                                img_meta['img_shape'], dtype=np.uint8)
                        imgs.append(img)
                else:
                    imgs = tensor2imgs(img_tensor,
                                       **img_metas[0]['img_norm_cfg'])
                for i, img in enumerate(imgs):
                    h, w, _ = img_metas[i]['img_shape']
                    img_show = img[:h, :w, :]
                    if out_dir:
                        out_file = osp.join(out_dir,
                                            img_metas[i]['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        gt_bboxes[i],
                        show=show,
                        out_file=out_file)
            else:
                img_tensor, img_metas, img_norm_cfg = \
                    retrieve_img_tensor_and_meta(data)

                if img_tensor.size(1) == 1:
                    imgs = tensor2grayimgs(img_tensor, **img_norm_cfg)
                else:
                    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
                assert len(imgs) == len(img_metas)

                for j, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    img_shape, ori_shape = img_meta['img_shape'], img_meta[
                        'ori_shape']
                    img_show = img[:img_shape[0], :img_shape[1]]
                    img_show = mmcv.imresize(img_show,
                                             (ori_shape[1], ori_shape[0]))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[j],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results
