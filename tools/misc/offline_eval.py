# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json

import mmengine
from mmengine import Config, DictAction
from mmengine.evaluator import Evaluator

from mmocr.registry import DATASETS
from mmocr.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Offline evaluation of the '
                                     'prediction saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument(
        'pkl_results', help='Path to the predictions in '
        'pickle format')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    register_all_modules()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    predictions = mmengine.load(args.pkl_results)

    evaluator = Evaluator(cfg.test_evaluator)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    eval_results = evaluator.offline_evaluate(dataset, predictions)
    print(json.dumps(eval_results))


if __name__ == '__main__':
    main()
