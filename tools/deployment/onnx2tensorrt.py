# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from typing import Iterable

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.tensorrt import is_tensorrt_plugin_loaded, onnx2trt, save_trt_engine
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from mmocr.core.deployment import (ONNXRuntimeDetector, ONNXRuntimeRecognizer,
                                   TensorRTDetector, TensorRTRecognizer)
from mmocr.datasets.pipelines.crop import crop_img  # noqa: F401
from mmocr.utils import is_2dlist


def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)


def _prepare_input_img(imgs, test_pipeline: Iterable[dict]):
    """Inference image(s) with the detector.

    Args:
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
        test_pipeline (Iterable[dict]): Test pipline of configuration.
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

    test_pipeline = replace_ImageToTensor(test_pipeline)
    test_pipeline = Compose(test_pipeline)

    data = []
    for img in imgs:
        # prepare data
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


def onnx2tensorrt(onnx_file: str,
                  model_type: str,
                  trt_file: str,
                  config: dict,
                  input_config: dict,
                  fp16: bool = False,
                  verify: bool = False,
                  show: bool = False,
                  workspace_size: int = 1,
                  verbose: bool = False):
    import tensorrt as trt
    min_shape = input_config['min_shape']
    max_shape = input_config['max_shape']
    # create trt engine and wrapper
    opt_shape_dict = {'input': [min_shape, min_shape, max_shape]}
    max_workspace_size = get_GiB(workspace_size)
    trt_engine = onnx2trt(
        onnx_file,
        opt_shape_dict,
        log_level=trt.Logger.VERBOSE if verbose else trt.Logger.ERROR,
        fp16_mode=fp16,
        max_workspace_size=max_workspace_size)
    save_dir, _ = osp.split(trt_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_trt_engine(trt_engine, trt_file)
    print(f'Successfully created TensorRT engine: {trt_file}')

    if verify:
        mm_inputs = _prepare_input_img(input_config['input_path'],
                                       config.data.test.pipeline)

        imgs = mm_inputs.pop('img')
        img_metas = mm_inputs.pop('img_metas')

        if isinstance(imgs, list):
            imgs = imgs[0]

        img_list = [img[None, :] for img in imgs]

        # Get results from ONNXRuntime
        if model_type == 'det':
            onnx_model = ONNXRuntimeDetector(onnx_file, config, 0)
        else:
            onnx_model = ONNXRuntimeRecognizer(onnx_file, config, 0)
        onnx_out = onnx_model.simple_test(
            img_list[0], img_metas[0], rescale=True)

        # Get results from TensorRT
        if model_type == 'det':
            trt_model = TensorRTDetector(trt_file, config, 0)
        else:
            trt_model = TensorRTRecognizer(trt_file, config, 0)
        img_list[0] = img_list[0].to(torch.device('cuda:0'))
        trt_out = trt_model.simple_test(
            img_list[0], img_metas[0], rescale=True)

        # compare results
        same_diff = 'same'
        if model_type == 'recog':
            for onnx_result, trt_result in zip(onnx_out, trt_out):
                if onnx_result['text'] != trt_result['text'] or \
                     not np.allclose(
                            np.array(onnx_result['score']),
                            np.array(trt_result['score']),
                            rtol=1e-4,
                            atol=1e-4):
                    same_diff = 'different'
                    break
        else:
            for onnx_result, trt_result in zip(onnx_out[0]['boundary_result'],
                                               trt_out[0]['boundary_result']):
                if not np.allclose(
                        np.array(onnx_result),
                        np.array(trt_result),
                        rtol=1e-4,
                        atol=1e-4):
                    same_diff = 'different'
                    break
        print('The outputs are {} between TensorRT and ONNX'.format(same_diff))

        if show:
            onnx_img = onnx_model.show_result(
                input_config['input_path'],
                onnx_out[0],
                out_file='onnx.jpg',
                show=False)
            trt_img = trt_model.show_result(
                input_config['input_path'],
                trt_out[0],
                out_file='tensorrt.jpg',
                show=False)
            if onnx_img is None:
                onnx_img = cv2.imread(input_config['input_path'])
            if trt_img is None:
                trt_img = cv2.imread(input_config['input_path'])

            cv2.imshow('TensorRT', trt_img)
            cv2.imshow('ONNXRuntime', onnx_img)
            cv2.waitKey()
    return


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMOCR models from ONNX to TensorRT')
    parser.add_argument('model_config', help='Config file of the model')
    parser.add_argument(
        'model_type',
        type=str,
        help='Detection or recognition model to deploy.',
        choices=['recog', 'det'])
    parser.add_argument('image_path', type=str, help='Image for test')
    parser.add_argument('onnx_file', help='Path to the input ONNX model')
    parser.add_argument(
        '--trt-file',
        type=str,
        help='Path to the output TensorRT engine',
        default='tmp.trt')
    parser.add_argument(
        '--max-shape',
        type=int,
        nargs=4,
        default=[1, 3, 400, 600],
        help='Maximum shape of model input.')
    parser.add_argument(
        '--min-shape',
        type=int,
        nargs=4,
        default=[1, 3, 400, 600],
        help='Minimum shape of model input.')
    parser.add_argument(
        '--workspace-size',
        type=int,
        default=1,
        help='Max workspace size in GiB.')
    parser.add_argument('--fp16', action='store_true', help='Enable fp16 mode')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Whether Verify the outputs of ONNXRuntime and TensorRT.',
        default=True)
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether visiualize outputs of ONNXRuntime and TensorRT.',
        default=True)
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether to verbose logging messages while creating \
                TensorRT engine.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    assert is_tensorrt_plugin_loaded(), 'TensorRT plugin should be compiled.'
    args = parse_args()

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

    # check arguments
    assert osp.exists(args.model_config), 'Config {} not found.'.format(
        args.model_config)
    assert osp.exists(args.onnx_file), \
        'ONNX model {} not found.'.format(args.onnx_file)
    assert args.workspace_size >= 0, 'Workspace size less than 0.'
    for max_value, min_value in zip(args.max_shape, args.min_shape):
        assert max_value >= min_value, \
            'max_shape should be larger than min shape'

    input_config = {
        'min_shape': args.min_shape,
        'max_shape': args.max_shape,
        'input_path': args.image_path
    }

    cfg = mmcv.Config.fromfile(args.model_config)
    if cfg.data.test.get('pipeline', None) is None:
        if is_2dlist(cfg.data.test.datasets):
            cfg.data.test.pipeline = \
                cfg.data.test.datasets[0][0].pipeline
        else:
            cfg.data.test.pipeline = \
                cfg.data.test['datasets'][0].pipeline
    if is_2dlist(cfg.data.test.pipeline):
        cfg.data.test.pipeline = cfg.data.test.pipeline[0]
    onnx2tensorrt(
        args.onnx_file,
        args.model_type,
        args.trt_file,
        cfg,
        input_config,
        fp16=args.fp16,
        verify=args.verify,
        show=args.show,
        workspace_size=args.workspace_size,
        verbose=args.verbose)
