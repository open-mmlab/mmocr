# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch


def tensor2grayimgs(tensor, mean=(127, ), std=(127, ), **kwargs):
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
