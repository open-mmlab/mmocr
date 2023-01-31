# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

from mmengine import Config
from mmocr.registry import MODELS
from mmocr.utils import register_all_modules

register_all_modules()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[640, 640],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape, please use --shape h w')

    input_shape = (1, 3, h, w)

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)

    flops = FlopCountAnalysis(model, torch.ones(input_shape))

    # params = parameter_count_table(model)
    flops_data = flop_count_table(flops)

    print(flops_data)

    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
