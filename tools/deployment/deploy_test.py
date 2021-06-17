import argparse

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info

from mmdet.apis import single_gpu_test
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.models.deploy_helper import (ONNXRuntimeDetector, ONNXRuntimeRecognizer, TensorRTDetector, TensorRTRecognizer)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMOCR test (and eval) a onnx or tensorrt model.')
    parser.add_argument(
        'model_config',
        type=str,
        help='Config file.')
    parser.add_argument(
        'model_file',
        type=str,
        help='Input file name for evaluation.')
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
    if args.device == 'cpu':
        args.device = None
    device = torch.device(args.device)

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
