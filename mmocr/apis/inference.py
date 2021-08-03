import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


def disable_text_recog_aug_test(cfg, set_types=None):
    """Remove aug_test from test pipeline of text recognition.
    Args:
        cfg (mmcv.Config): Input config.
        set_types (list[str]): Type of dataset source. Should be
            None or sublist of ['test', 'val']

    Returns:
        cfg (mmcv.Config): Output config removing
            `MultiRotateAugOCR` in test pipeline.
    """
    assert set_types is None or isinstance(set_types, list)
    if set_types is None:
        set_types = ['val', 'test']
    for set_type in set_types:
        if cfg.data[set_type].pipeline[1].type == 'MultiRotateAugOCR':
            cfg.data[set_type].pipeline = [
                cfg.data[set_type].pipeline[0],
                *cfg.data[set_type].pipeline[1].transforms
            ]

    return cfg


def model_inference(model, imgs, batch_mode=False):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
        batch_mode (bool): If True, use batch mode for inference.
    Returns:
        result (dict): Predicted results.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
        if not isinstance(imgs[0], (np.ndarray, str)):
            raise AssertionError('imgs must be strings or numpy arrays')

    elif isinstance(imgs, (np.ndarray, str)):
        imgs = [imgs]
        is_batch = False
    else:
        raise AssertionError('imgs must be strings or numpy arrays')

    is_ndarray = isinstance(imgs[0], np.ndarray)

    cfg = model.cfg

    if batch_mode:
        cfg = disable_text_recog_aug_test(cfg, set_types=['test'])

    device = next(model.parameters()).device  # model device

    if is_ndarray:
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromNdarray'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if is_ndarray:
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)

        # build the data pipeline
        data = test_pipeline(data)
        # get tensor from list to stack for batch mode (text detection)
        if batch_mode:
            if cfg.data.test.pipeline[1].type == 'MultiScaleFlipAug':
                for key, value in data.items():
                    data[key] = value[0]
        datas.append(data)

    if isinstance(datas[0]['img'], list) and len(datas) > 1:
        raise Exception('aug test does not support '
                        f'inference with batch size '
                        f'{len(datas)}')

    data = collate(datas, samples_per_gpu=len(imgs))

    # process img_metas
    if isinstance(data['img_metas'], list):
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
    else:
        data['img_metas'] = data['img_metas'].data

    if isinstance(data['img'], list):
        data['img'] = [img.data for img in data['img']]
        if isinstance(data['img'][0], list):
            data['img'] = [img[0] for img in data['img']]
    else:
        data['img'] = data['img'].data

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


def text_model_inference(model, input_sentence):
    """Inference text(s) with the entity recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        input_sentence (str): A text entered by the user.

    Returns:
        result (dict): Predicted results.
    """

    assert isinstance(input_sentence, str)

    cfg = model.cfg
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = {'text': input_sentence, 'label': {}}

    # build the data pipeline
    data = test_pipeline(data)
    if isinstance(data['img_metas'], dict):
        img_metas = data['img_metas']
    else:
        img_metas = data['img_metas'].data

    assert isinstance(img_metas, dict)
    img_metas = {
        'input_ids': img_metas['input_ids'].unsqueeze(0),
        'attention_masks': img_metas['attention_masks'].unsqueeze(0),
        'token_type_ids': img_metas['token_type_ids'].unsqueeze(0),
        'labels': img_metas['labels'].unsqueeze(0)
    }
    # forward the model
    with torch.no_grad():
        result = model(None, img_metas, return_loss=False)
    return result
