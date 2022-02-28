# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info
from mmdet.apis import single_gpu_test

from mmocr.apis.inference import disable_text_recog_aug_test
from mmocr.core.deployment import (ONNXRuntimeDetector, ONNXRuntimeRecognizer,
                                   TensorRTDetector, TensorRTRecognizer)
from mmocr.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMOCR test (and eval) a onnx or tensorrt model.')
    parser.add_argument('model_config', type=str, help='Config file.')
    parser.add_argument(
        'model_file', type=str, help='Input file name for evaluation.')
    parser.add_argument(
        'model_type',
        type=str,
        help='Detection or recognition model to deploy.',
        choices=['recog', 'det'])
    parser.add_argument(
        'backend',
        type=str,
        help='Which backend to test, TensorRT or ONNXRuntime.',
        choices=['TensorRT', 'ONNXRuntime'])
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='The evaluation metrics, which depends on the dataset, e.g.,'
        '"bbox", "seg", "proposal" for COCO, and "mAP", "recall" for'
        'PASCAL VOC.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')

    args = parser.parse_args()

    return args


def main():
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

    if args.device == 'cpu':
        args.device = None

    cfg = Config.fromfile(args.model_config)

    # build the model
    if args.model_type == 'det':
        if args.backend == 'TensorRT':
            model = TensorRTDetector(args.model_file, cfg, 0)
        else:
            model = ONNXRuntimeDetector(args.model_file, cfg, 0)
    else:
        if args.backend == 'TensorRT':
            model = TensorRTRecognizer(args.model_file, cfg, 0)
        else:
            model = ONNXRuntimeRecognizer(args.model_file, cfg, 0)

    # build the dataloader
    samples_per_gpu = 1
    cfg = disable_text_recog_aug_test(cfg)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {}
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
