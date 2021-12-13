# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.image import tensor2imgs
from mmcv.parallel import DataContainer
from mmdet.core import encode_mask_results


def tensor2grayimgs(tensor, mean=(127), std=(127), **kwargs):
    """Convert tensor to 1-channel gray images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W).
        mean (tuple[float], optional): Mean of images. Defaults to (127).
        std (tuple[float], optional): Standard deviation of images.
            Defaults to (127).

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    assert torch.is_tensor(tensor) and tensor.ndim == 4
    assert tensor.size(1) == len(mean) == len(std) == 1

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(img, mean, std, to_bgr=False).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def retrieval_img_tensor_and_meta(data):
    if isinstance(data['img'], torch.Tensor):
        # for textrecog with batch_size > 1
        # and not use 'DefaultFormatBundle' in pipeline
        img_tensor = data['img']
        img_metas = data['img_metas'].data[0]
    elif isinstance(data['img'], list):
        if isinstance(data['img'][0], torch.Tensor):
            # for textrecog with aug_test and batch_size = 1
            img_tensor = data['img'][0]
        elif isinstance(data['img'][0], DataContainer):
            # for textdet with 'MultiScaleFlipAug'
            # and 'DefaultFormatBundle' in pipeline
            img_tensor = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
    elif isinstance(data['img'], DataContainer):
        # for textrecog with 'DefaultFormatBundle' in pipeline
        img_tensor = data['img'].data[0]
        img_metas = data['img_metas'].data[0]

    must_keys = ['img_norm_cfg', 'ori_filename', 'img_shape', 'ori_shape']
    for key in must_keys:
        if key not in img_metas[0]:
            raise KeyError(
                f'Please add {key} to the "meta_keys" in the pipeline')

    img_norm_cfg = img_metas[0]['img_norm_cfg']
    if max(img_norm_cfg['mean']) <= 1:
        img_norm_cfg['mean'] = [255 * x for x in img_norm_cfg['mean']]
        img_norm_cfg['std'] = [255 * x for x in img_norm_cfg['std']]

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
    for i, data in enumerate(data_loader):
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
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
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
                    retrieval_img_tensor_and_meta(data)

                if img_tensor.size(1) == 1:
                    imgs = tensor2grayimgs(img_tensor, **img_norm_cfg)
                else:
                    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    if img_tensor.size(1) == 3:
                        h, w, _ = img_meta['img_shape']
                        img_show = img[:h, :w, :]
                        ori_h, ori_w = img_meta['ori_shape'][:-1]
                    elif img_tensor.size(1) == 1:
                        h, w = img_meta['img_shape']
                        img_show = img[:h, :w]
                        ori_h, ori_w = img_meta['ori_shape']

                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
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
