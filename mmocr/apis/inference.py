import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.datasets import replace_ImageToTensor


def model_inference(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray): Image files.

    Returns:
        result (dict): Detection results.
    """

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(img, np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
    
    # build the data pipeline
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    # process img_metas
    if isinstance(data['img_metas'], list):
        data['img_metas'] = data['img_metas'][0].data
    else:
        data['img_metas'] = data['img_metas'].data[0]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]
    return result
