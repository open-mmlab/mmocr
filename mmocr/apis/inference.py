import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose


def model_inference(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str): Image files.

    Returns:
        result (dict): Detection results.
    """
    assert isinstance(img, str)

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    # process img_metas
    data['img_metas'] = data['img_metas'][0].data

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
