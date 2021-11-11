#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import time
from argparse import ArgumentParser
from itertools import compress

import mmcv
from mmcv.utils import ProgressBar

from mmocr.apis import init_detector, model_inference
from mmocr.core.evaluation.ocr_metric import eval_ocr_metric
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401
from mmocr.utils import get_root_logger, list_from_file, list_to_file


def save_results(img_paths, pred_labels, gt_labels, res_dir):
    """Save predicted results to txt file.

    Args:
        img_paths (list[str])
        pred_labels (list[str])
        gt_labels (list[str])
        res_dir (str)
    """
    assert len(img_paths) == len(pred_labels) == len(gt_labels)
    corrects = [pred == gt for pred, gt in zip(pred_labels, gt_labels)]
    wrongs = [not c for c in corrects]
    lines = [
        f'{img} {pred} {gt}'
        for img, pred, gt in zip(img_paths, pred_labels, gt_labels)
    ]
    list_to_file(osp.join(res_dir, 'results.txt'), lines)
    list_to_file(osp.join(res_dir, 'correct.txt'), compress(lines, corrects))
    list_to_file(osp.join(res_dir, 'wrong.txt'), compress(lines, wrongs))


def main():
    parser = ArgumentParser()
    parser.add_argument('img_root_path', type=str, help='Image root path')
    parser.add_argument('img_list', type=str, help='Image path list file')
    parser.add_argument('config', type=str, help='Config file')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument(
        '--out_dir', type=str, default='./results', help='Dir to save results')
    parser.add_argument(
        '--show', action='store_true', help='show image or save')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    args = parser.parse_args()

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.out_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if hasattr(model, 'module'):
        model = model.module
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    # Start Inference
    out_vis_dir = osp.join(args.out_dir, 'out_vis_dir')
    mmcv.mkdir_or_exist(out_vis_dir)
    correct_vis_dir = osp.join(args.out_dir, 'correct')
    mmcv.mkdir_or_exist(correct_vis_dir)
    wrong_vis_dir = osp.join(args.out_dir, 'wrong')
    mmcv.mkdir_or_exist(wrong_vis_dir)
    img_paths, pred_labels, gt_labels = [], [], []
    total_img_num = sum([1 for _ in open(args.img_list)])
    progressbar = ProgressBar(task_num=total_img_num)
    num_gt_label = 0
    for line in list_from_file(args.img_list):
        progressbar.update()
        item_list = line.strip().split()
        img_file = item_list[0]
        gt_label = ''
        if len(item_list) >= 2:
            gt_label = item_list[1]
            num_gt_label += 1
        img_path = osp.join(args.img_root_path, img_file)
        if not osp.exists(img_path):
            raise FileNotFoundError(img_path)
        # Test a single image
        result = model_inference(model, img_path)
        pred_label = result['text']

        out_img_name = '_'.join(img_file.split('/'))
        out_file = osp.join(out_vis_dir, out_img_name)
        kwargs_dict = {
            'gt_label': gt_label,
            'show': args.show,
            'out_file': '' if args.show else out_file
        }
        model.show_result(img_path, result, **kwargs_dict)
        if gt_label != '':
            if gt_label == pred_label:
                dst_file = osp.join(correct_vis_dir, out_img_name)
            else:
                dst_file = osp.join(wrong_vis_dir, out_img_name)
            shutil.copy(out_file, dst_file)
        img_paths.append(img_path)
        gt_labels.append(gt_label)
        pred_labels.append(pred_label)

    # Save results
    save_results(img_paths, pred_labels, gt_labels, args.out_dir)

    if num_gt_label == len(pred_labels):
        # eval
        eval_results = eval_ocr_metric(pred_labels, gt_labels)
        logger.info('\n' + '-' * 100)
        info = ('eval on testset with img_root_path '
                f'{args.img_root_path} and img_list {args.img_list}\n')
        logger.info(info)
        logger.info(eval_results)

    print(f'\nInference done, and results saved in {args.out_dir}\n')


if __name__ == '__main__':
    main()
