#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from argparse import ArgumentParser

import mmcv
from mmcv.utils import ProgressBar
from mmdet.apis import inference_detector, init_detector

from mmocr.models import build_detector  # noqa: F401
from mmocr.utils import list_from_file, list_to_file


def gen_target_path(target_root_path, src_name, suffix):
    """Gen target file path.

    Args:
        target_root_path (str): The target root path.
        src_name (str): The source file name.
        suffix (str): The suffix of target file.
    """
    assert isinstance(target_root_path, str)
    assert isinstance(src_name, str)
    assert isinstance(suffix, str)

    file_name = osp.split(src_name)[-1]
    name = osp.splitext(file_name)[0]
    return osp.join(target_root_path, name + suffix)


def save_results(result, out_dir, img_name, score_thr=0.3):
    """Save result of detected bounding boxes (quadrangle or polygon) to txt
    file.

    Args:
        result (dict): Text Detection result for one image.
        img_name (str): Image file name.
        out_dir (str): Dir of txt files to save detected results.
        score_thr (float, optional): Score threshold to filter bboxes.
    """
    assert 'boundary_result' in result
    assert score_thr > 0 and score_thr < 1

    txt_file = gen_target_path(out_dir, img_name, '.txt')
    valid_boundary_res = [
        res for res in result['boundary_result'] if res[-1] > score_thr
    ]
    lines = [
        ','.join([str(round(x)) for x in row]) for row in valid_boundary_res
    ]
    list_to_file(txt_file, lines)


def main():
    parser = ArgumentParser()
    parser.add_argument('img_root', type=str, help='Image root path')
    parser.add_argument('img_list', type=str, help='Image path list file')
    parser.add_argument('config', type=str, help='Config file')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='Bbox score threshold')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./results',
        help='Dir to save '
        'visualize images '
        'and bbox')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    args = parser.parse_args()

    assert 0 < args.score_thr < 1

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
    out_txt_dir = osp.join(args.out_dir, 'out_txt_dir')
    mmcv.mkdir_or_exist(out_txt_dir)

    total_img_num = sum([1 for _ in open(args.img_list)])
    progressbar = ProgressBar(task_num=total_img_num)
    for line in list_from_file(args.img_list):
        progressbar.update()
        img_path = osp.join(args.img_root, line.strip())
        if not osp.exists(img_path):
            raise FileNotFoundError(img_path)
        # Test a single image
        result = inference_detector(model, img_path)
        img_name = osp.basename(img_path)
        # save result
        save_results(result, out_txt_dir, img_name, score_thr=args.score_thr)
        # show result
        out_file = osp.join(out_vis_dir, img_name)
        kwargs_dict = {
            'score_thr': args.score_thr,
            'show': False,
            'out_file': out_file
        }
        model.show_result(img_path, result, **kwargs_dict)

    print(f'\nInference done, and results saved in {args.out_dir}\n')


if __name__ == '__main__':
    main()
