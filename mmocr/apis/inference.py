import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


def model_inference(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.

    Returns:
        result (dict): Detection results.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    elif isinstance(imgs, (np.ndarray, str)):
        imgs = [imgs]
        is_batch = False
    else:
        raise AssertionError('imgs must be strings or numpy arrays')

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)

        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))

    # process img_metas
    if isinstance(data['img_metas'], list):
        if len(data['img_metas']) == 1:
            data['img_metas'] = data['img_metas'][0].data
        else:
            data['img_metas'] = [
                img_metas.data[0] for img_metas in data['img_metas']
            ]
    else:
        data['img_metas'] = data['img_metas'].data[0]

    # process img
    if isinstance(img, np.ndarray):
        data['img'] = [img.data[0] for img in data['img']]
        
    for idx, img in enumerate(data['img']):
        if img.dim() == 3:
            data['img'][idx] = img.unsqueeze(0)

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
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results
