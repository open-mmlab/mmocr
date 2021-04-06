import codecs
import os.path as osp
from argparse import ArgumentParser

import mmcv
import numpy as np
import torch
from mmcv.utils import ProgressBar

from mmdet.apis import inference_detector, init_detector
from mmocr.core.evaluation.utils import filter_result
from mmocr.models import build_detector  # noqa: F401


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

    dir_name, file_name = osp.split(src_name)
    name, file_suffix = osp.splitext(file_name)
    return target_root_path + '/' + name + suffix


def save_2darray(mat, file_name):
    """Save 2d array to txt file.

    Args:
        mat (ndarray): 2d-array of shape (n, m).
        file_name (str): The output file name.
    """
    with codecs.open(file_name, 'w', 'utf-8') as fw:
        for row in mat:
            row_str = ','.join([str(x) for x in row])
            fw.write(row_str + '\n')


def save_bboxes_quadrangles(bboxes_with_scores,
                            quadrangles_with_scores,
                            img_name,
                            out_bbox_txt_dir,
                            out_quadrangle_txt_dir,
                            score_thr=0.3,
                            save_score=True):
    """Save results of detected bounding boxes and quadrangles to txt file.

    Args:
        bboxes_with_scores (ndarray): Detected bboxes of shape (n,5).
        quadrangles_with_scores (ndarray): Detected quadrangles of shape (n,9).
        img_name (str): Image file name.
        out_bbox_txt_dir (str): Dir of txt files to save detected bboxes
                                results.
        out_quadrangle_txt_dir (str): Dir of txt files to save
                                quadrangle results.
        score_thr (float, optional): Score threshold for bboxes.
        save_score (bool, optional): Whether to save score at each line end
        to search best threshold when evaluating.
    """
    assert bboxes_with_scores.ndim == 2
    assert bboxes_with_scores.shape[1] == 5 or bboxes_with_scores.shape[1] == 9
    assert quadrangles_with_scores.ndim == 2
    assert quadrangles_with_scores.shape[1] == 9
    assert bboxes_with_scores.shape[0] >= quadrangles_with_scores.shape[0]
    assert isinstance(img_name, str)
    assert isinstance(out_bbox_txt_dir, str)
    assert isinstance(out_quadrangle_txt_dir, str)
    assert isinstance(score_thr, float)
    assert score_thr >= 0 and score_thr < 1

    # filter out invalid results
    initial_valid_bboxes, valid_bbox_scores = filter_result(
        bboxes_with_scores[:, :-1], bboxes_with_scores[:, -1], score_thr)
    if initial_valid_bboxes.shape[1] == 4:
        valid_bboxes = np.ndarray(
            (initial_valid_bboxes.shape[0], 8)).astype(int)
        idx_list = [0, 1, 2, 1, 2, 3, 0, 3]
        for i in range(8):
            valid_bboxes[:, i] = initial_valid_bboxes[:, idx_list[i]]

    elif initial_valid_bboxes.shape[1] == 8:
        valid_bboxes = initial_valid_bboxes

    valid_quadrangles, valid_quadrangle_scores = filter_result(
        quadrangles_with_scores[:, :-1], quadrangles_with_scores[:, -1],
        score_thr)

    # gen target file path
    bbox_txt_file = gen_target_path(out_bbox_txt_dir, img_name, '.txt')
    quadrangle_txt_file = gen_target_path(out_quadrangle_txt_dir, img_name,
                                          '.txt')

    # save txt
    if save_score:
        valid_bboxes = np.concatenate(
            (valid_bboxes, valid_bbox_scores.reshape(-1, 1)), axis=1)
        valid_quadrangles = np.concatenate(
            (valid_quadrangles, valid_quadrangle_scores.reshape(-1, 1)),
            axis=1)

    save_2darray(valid_bboxes, bbox_txt_file)
    save_2darray(valid_quadrangles, quadrangle_txt_file)


def main():
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='Config file')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('img_root', type=str, help='Image root path')
    parser.add_argument('img_list', type=str, help='Image path list file')

    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='Bbox score threshold')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./results',
        help='Dir to save '
        'visualize images '
        'and bbox')
    args = parser.parse_args()

    assert args.score_thr > 0 and args.score_thr < 1

    # build the model from a config file and a checkpoint file
    device = 'cuda:' + str(torch.cuda.current_device())
    model = init_detector(args.config, args.checkpoint, device=device)
    if hasattr(model, 'module'):
        model = model.module
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    # Start Inference
    out_vis_dir = osp.join(args.out_dir, 'out_vis_dir')
    mmcv.mkdir_or_exist(out_vis_dir)

    total_img_num = sum([1 for _ in open(args.img_list)])
    progressbar = ProgressBar(task_num=total_img_num)
    with codecs.open(args.img_list, 'r', 'utf-8') as fr:
        for line in fr:
            progressbar.update()
            img_path = args.img_root + '/' + line.strip()
            if not osp.exists(img_path):
                raise FileNotFoundError(img_path)
            # Test a single image
            result = inference_detector(model, img_path)
            img_name = osp.basename(img_path)
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
