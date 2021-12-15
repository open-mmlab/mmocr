# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from functools import partial

import cv2
import numpy as np
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.parallel import collate
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from torch import nn

from mmocr.apis import init_detector
from mmocr.core.deployment import ONNXRuntimeDetector, ONNXRuntimeRecognizer
from mmocr.datasets.pipelines.crop import crop_img  # noqa: F401
from mmocr.utils import is_2dlist


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def _prepare_data(cfg, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
    Returns:
        result (dict): Predicted results.
    """
    if isinstance(imgs, (list, tuple)):
        if not isinstance(imgs[0], (np.ndarray, str)):
            raise AssertionError('imgs must be strings or numpy arrays')

    elif isinstance(imgs, (np.ndarray, str)):
        imgs = [imgs]
    else:
        raise AssertionError('imgs must be strings or numpy arrays')

    is_ndarray = isinstance(imgs[0], np.ndarray)

    if is_ndarray:
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromNdarray'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    data = []
    for img in imgs:
        # prepare data
        if is_ndarray:
            # directly add img
            datum = dict(img=img)
        else:
            # add information into dict
            datum = dict(img_info=dict(filename=img), img_prefix=None)

        # build the data pipeline
        datum = test_pipeline(datum)
        # get tensor from list to stack for batch mode (text detection)
        data.append(datum)

    if isinstance(data[0]['img'], list) and len(data) > 1:
        raise Exception('aug test does not support '
                        f'inference with batch size '
                        f'{len(data)}')

    data = collate(data, samples_per_gpu=len(imgs))

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
    return data


def pytorch2onnx(model: nn.Module,
                 model_type: str,
                 img_path: str,
                 verbose: bool = False,
                 show: bool = False,
                 opset_version: int = 11,
                 output_file: str = 'tmp.onnx',
                 verify: bool = False,
                 dynamic_export: bool = False,
                 device_id: int = 0):
    """Export PyTorch model to ONNX model and verify the outputs are same
    between PyTorch and ONNX.

    Args:
        model (nn.Module): PyTorch model we want to export.
        model_type (str): Model type, detection or recognition model.
        img_path (str): We need to use this input to execute the model.
        opset_version (int): The onnx op version. Default: 11.
        verbose (bool): Whether print the computation graph. Default: False.
        show (bool): Whether visialize final results. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between PyTorch and ONNX.
            Default: False.
        dynamic_export (bool): Whether apply dynamic export.
            Default: False.
        device_id (id): Device id to place model and data.
            Default: 0
    """
    device = torch.device(type='cuda', index=device_id)
    model.to(device).eval()
    _convert_batchnorm(model)

    # prepare inputs
    mm_inputs = _prepare_data(cfg=model.cfg, imgs=img_path)
    imgs = mm_inputs.pop('img')
    img_metas = mm_inputs.pop('img_metas')

    if isinstance(imgs, list):
        imgs = imgs[0]

    img_list = [img[None, :].to(device) for img in imgs]

    origin_forward = model.forward
    if (model_type == 'det'):
        model.forward = partial(
            model.simple_test, img_metas=img_metas, rescale=True)
    else:
        model.forward = partial(
            model.forward,
            img_metas=img_metas,
            return_loss=False,
            rescale=True)

    # pytorch has some bug in pytorch1.3, we have to fix it
    # by replacing these existing op
    register_extra_symbolics(opset_version)
    dynamic_axes = None
    if dynamic_export and model_type == 'det':
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'output': {
                0: 'batch',
                2: 'height',
                3: 'width'
            }
        }
    elif dynamic_export and model_type == 'recog':
        dynamic_axes = {
            'input': {
                0: 'batch',
                3: 'width'
            },
            'output': {
                0: 'batch',
                1: 'seq_len',
                2: 'num_classes'
            }
        }
    with torch.no_grad():
        torch.onnx.export(
            model, (img_list[0], ),
            output_file,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=verbose,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes)
    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        scale_factor = (0.5, 0.5) if model_type == 'det' else (1, 0.5)
        if dynamic_export:
            # scale image for dynamic shape test
            img_list = [
                nn.functional.interpolate(_, scale_factor=scale_factor)
                for _ in img_list
            ]
            if model_type == 'det':
                img_metas[0][0][
                    'scale_factor'] = img_metas[0][0]['scale_factor'] * (
                        scale_factor * 2)

        # check the numerical value
        # get pytorch output
        with torch.no_grad():
            model.forward = origin_forward
            pytorch_out = model.simple_test(
                img_list[0], img_metas[0], rescale=True)

        # get onnx output
        if model_type == 'det':
            onnx_model = ONNXRuntimeDetector(output_file, model.cfg, device_id)
        else:
            onnx_model = ONNXRuntimeRecognizer(output_file, model.cfg,
                                               device_id)
        onnx_out = onnx_model.simple_test(
            img_list[0], img_metas[0], rescale=True)

        # compare results
        same_diff = 'same'
        if model_type == 'recog':
            for onnx_result, pytorch_result in zip(onnx_out, pytorch_out):
                if onnx_result['text'] != pytorch_result[
                        'text'] or not np.allclose(
                            np.array(onnx_result['score']),
                            np.array(pytorch_result['score']),
                            rtol=1e-4,
                            atol=1e-4):
                    same_diff = 'different'
                    break
        else:
            for onnx_result, pytorch_result in zip(
                    onnx_out[0]['boundary_result'],
                    pytorch_out[0]['boundary_result']):
                if not np.allclose(
                        np.array(onnx_result),
                        np.array(pytorch_result),
                        rtol=1e-4,
                        atol=1e-4):
                    same_diff = 'different'
                    break
        print('The outputs are {} between PyTorch and ONNX'.format(same_diff))

        if show:
            onnx_img = onnx_model.show_result(
                img_path, onnx_out[0], out_file='onnx.jpg', show=False)
            pytorch_img = model.show_result(
                img_path, pytorch_out[0], out_file='pytorch.jpg', show=False)
            if onnx_img is None:
                onnx_img = cv2.imread(img_path)
            if pytorch_img is None:
                pytorch_img = cv2.imread(img_path)

            cv2.imshow('PyTorch', pytorch_img)
            cv2.imshow('ONNXRuntime', onnx_img)
            cv2.waitKey()
    return


def main():
    parser = ArgumentParser(
        description='Convert MMOCR models from pytorch to ONNX')
    parser.add_argument('model_config', type=str, help='Config file.')
    parser.add_argument(
        'model_ckpt', type=str, help='Checkpint file (local or url).')
    parser.add_argument(
        'model_type',
        type=str,
        help='Detection or recognition model to deploy.',
        choices=['recog', 'det'])
    parser.add_argument('image_path', type=str, help='Input Image file.')
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file name of the onnx model.',
        default='tmp.onnx')
    parser.add_argument(
        '--device-id', default=0, help='Device used for inference.')
    parser.add_argument(
        '--opset-version',
        type=int,
        help='ONNX opset version, default to 11.',
        default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Whether verify the outputs of onnx and pytorch are same.',
        default=False)
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether print the computation graph.',
        default=False)
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether visualize final output.',
        default=False)
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether dynamically export onnx model.',
        default=False)
    args = parser.parse_args()

    device = torch.device(type='cuda', index=args.device_id)

    # build model
    model = init_detector(args.model_config, args.model_ckpt, device=device)
    if hasattr(model, 'module'):
        model = model.module
    if model.cfg.data.test.get('pipeline', None) is None:
        model.cfg.data.test.pipeline = \
            model.cfg.data.test['datasets'][0].pipeline
    if is_2dlist(model.cfg.data.test.pipeline):
        model.cfg.data.test.pipeline = model.cfg.data.test.pipeline[0]

    pytorch2onnx(
        model,
        model_type=args.model_type,
        output_file=args.output_file,
        img_path=args.image_path,
        opset_version=args.opset_version,
        verify=args.verify,
        verbose=args.verbose,
        show=args.show,
        device_id=args.device_id,
        dynamic_export=args.dynamic_export)


if __name__ == '__main__':
    main()
