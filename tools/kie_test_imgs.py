#!/usr/bin/env python
import argparse
import ast
import os
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmocr.datasets import build_dataloader, build_dataset
from mmocr.models import build_detector


def test(model, data_loader, show=False, out_dir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            img_tensor = data['img'].data[0]
            img_metas = data['img_metas'].data[0]
            if np.prod(img_tensor.shape) == 0:
                imgs = [mmcv.imread(m['filename']) for m in img_metas]
            else:
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            gt_bboxes = [data['gt_bboxes'].data[0][0].numpy().tolist()]

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                if 'img_shape' in img_meta:
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]
                else:
                    img_show = img

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    gt_bboxes[i],
                    show=show,
                    out_file=out_file)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMOCR visualize for kie model.')
    parser.add_argument('config', help='Test config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument('--show', action='store_true', help='Show results.')
    parser.add_argument(
        '--show-dir', help='Directory where the output images will be saved.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        help='Use int or int list for gpu. Default is cpu',
        default=None)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    assert args.show or args.show_dir, ('Please specify at least one '
                                        'operation (show the results / save )'
                                        'the results with the argument '
                                        '"--show" or "--show-dir".')
    device = args.device
    if device is not None:
        device = ast.literal_eval(f'[{device}]')
    cfg = Config.fromfile(args.config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model, device_ids=device)
    test(model, data_loader, args.show, args.show_dir)


if __name__ == '__main__':
    main()
