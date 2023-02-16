# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import sys
from typing import Optional, Tuple

import cv2
import mmcv
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from mmengine.visualization import Visualizer

from mmocr.registry import DATASETS, VISUALIZERS


# TODO: Support for printing the change in key of results
def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='Path to model or dataset config.')
    parser.add_argument(
        '--phase',
        '-p',
        default='train',
        type=str,
        help='Phase of dataset to visualize. Use "train", "test" or "val" if '
        "you just want to visualize the default split. It's also possible to "
        'be a dataset variable name, which might be useful when a dataset '
        'split has multiple variants in the config.')
    parser.add_argument(
        '--mode',
        '-m',
        default='transformed',
        type=str,
        choices=['original', 'transformed', 'pipeline'],
        help='Display mode: display original pictures or '
        'transformed pictures or comparison pictures. "original" '
        'only visualizes the original dataset & annotations; '
        '"transformed" shows the resulting images processed through all the '
        'transforms; "pipeline" shows all the intermediate images. '
        'Defaults to "transformed".')
    parser.add_argument(
        '--output-dir',
        '-o',
        default=None,
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument(
        '--task',
        '-t',
        default='auto',
        choices=['auto', 'textdet', 'textrecog'],
        type=str,
        help='Specify the task type of the dataset. If "auto", the task type '
        'will be inferred from the config. If the script is unable to infer '
        'the task type, you need to specify it manually. Defaults to "auto".')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-number',
        '-n',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, '
        'must bigger than 0. if the number is bigger than length '
        'of dataset, show all the images in dataset; '
        'default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--show-interval',
        '-i',
        type=float,
        default=3,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def _get_adaptive_scale(img_shape: Tuple[int, int],
                        min_scale: float = 0.3,
                        max_scale: float = 3.0) -> float:
    """Get adaptive scale according to image shape.

    The target scale depends on the the short edge length of the image. If the
    short edge length equals 224, the output is 1.0. And output linear
    scales according the short edge length. You can also specify the minimum
    scale and the maximum scale to limit the linear scale.

    Args:
        img_shape (Tuple[int, int]): The shape of the canvas image.
        min_scale (int): The minimum scale. Defaults to 0.3.
        max_scale (int): The maximum scale. Defaults to 3.0.

    Returns:
        int: The adaptive scale.
    """
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.
    return min(max(scale, min_scale), max_scale)


def make_grid(imgs, names):
    """Concat list of pictures into a single big picture, align height here."""
    visualizer = Visualizer.get_current_instance()
    ori_shapes = [img.shape[:2] for img in imgs]
    max_height = int(max(img.shape[0] for img in imgs) * 1.1)
    min_width = min(img.shape[1] for img in imgs)
    horizontal_gap = min_width // 10
    img_scale = _get_adaptive_scale((max_height, min_width))

    texts = []
    text_positions = []
    start_x = 0
    for i, img in enumerate(imgs):
        pad_height = (max_height - img.shape[0]) // 2
        pad_width = horizontal_gap // 2
        # make border
        imgs[i] = cv2.copyMakeBorder(
            img,
            pad_height,
            max_height - img.shape[0] - pad_height + int(img_scale * 30 * 2),
            pad_width,
            pad_width,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255))
        texts.append(f'{"execution: "}{i}\n{names[i]}\n{ori_shapes[i]}')
        text_positions.append(
            [start_x + img.shape[1] // 2 + pad_width, max_height])
        start_x += img.shape[1] + horizontal_gap

    display_img = np.concatenate(imgs, axis=1)
    visualizer.set_image(display_img)
    img_scale = _get_adaptive_scale(display_img.shape[:2])
    visualizer.draw_texts(
        texts,
        positions=np.array(text_positions),
        font_sizes=img_scale * 7,
        colors='black',
        horizontal_alignments='center',
        font_families='monospace')
    return visualizer.get_image()


class InspectCompose(Compose):
    """Compose multiple transforms sequentially.

    And record "img" field of all results in one list.
    """

    def __init__(self, transforms, intermediate_imgs):
        super().__init__(transforms=transforms)
        self.intermediate_imgs = intermediate_imgs

    def __call__(self, data):
        if 'img' in data:
            self.intermediate_imgs.append({
                'name': 'original',
                'img': data['img'].copy()
            })
        self.ptransforms = [
            self.transforms[i] for i in range(len(self.transforms) - 1)
        ]
        for t in self.ptransforms:
            data = t(data)
            # Keep the same meta_keys in the PackDetInputs
            self.transforms[-1].meta_keys = [key for key in data]
            data_sample = self.transforms[-1](data)
            if data is None:
                return None
            if 'img' in data:
                self.intermediate_imgs.append({
                    'name':
                    t.__class__.__name__,
                    'dataset_sample':
                    data_sample['data_samples']
                })
        return data


def infer_dataset_task(task: str,
                       dataset_cfg: Config,
                       var_name: Optional[str] = None) -> str:
    """Try to infer the dataset's task type from the config and the variable
    name."""
    if task != 'auto':
        return task

    if dataset_cfg.pipeline is not None:
        if dataset_cfg.pipeline[-1].type == 'PackTextDetInputs':
            return 'textdet'
        elif dataset_cfg.pipeline[-1].type == 'PackTextRecogInputs':
            return 'textrecog'

    if var_name is not None:
        if 'det' in var_name:
            return 'textdet'
        elif 'rec' in var_name:
            return 'textrecog'

    raise ValueError(
        'Unable to infer the task type from dataset pipeline '
        'or variable name. Please specify the task type with --task argument '
        'explicitly.')


def obtain_dataset_cfg(cfg: Config, phase: str, mode: str, task: str) -> Tuple:
    """Obtain dataset and visualizer from config. Two modes are supported:
    1. Model Config Mode:
        In this mode, the input config should be a complete model config, which
        includes a dataset within pipeline and a visualizer.
    2. Dataset Config Mode:
        In this mode, the input config should be a complete dataset config,
        which only includes basic dataset information, and it may does not
        contain a visualizer and dataset pipeline.

    Examples:
        Typically, the model config files are stored in
        `configs/textdet/dbnet/xxx.py` and should look like:
        >>> train_dataloader = dict(
        >>>     batch_size=16,
        >>>     num_workers=8,
        >>>     persistent_workers=True,
        >>>     sampler=dict(type='DefaultSampler', shuffle=True),
        >>>     dataset=icdar2015_textdet_train)

        while the dataset config files are stored in
        `configs/textdet/_base_/datasets/xxx.py` and should be like:
        >>> icdar2015_textdet_train = dict(
        >>>     type='OCRDataset',
        >>>     data_root=ic15_det_data_root,
        >>>     ann_file='textdet_train.json',
        >>>     filter_cfg=dict(filter_empty_gt=True, min_size=32),
        >>>     pipeline=None)

    Args:
        cfg (Config): Config object.
        phase (str): The dataset phase to visualize.
        mode (str): Script mode.
        task (str): The current task type.

    Returns:
        Tuple: Tuple of (dataset, visualizer).
    """
    default_cfgs = dict(
        textdet=dict(
            visualizer=dict(
                type='TextDetLocalVisualizer',
                name='visualizer',
                vis_backends=[dict(type='LocalVisBackend')]),
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk'),
                    color_type='color_ignore_orientation'),
                dict(
                    type='LoadOCRAnnotations',
                    with_polygon=True,
                    with_bbox=True,
                    with_label=True,
                ),
                dict(
                    type='PackTextDetInputs',
                    meta_keys=('img_path', 'ori_shape', 'img_shape'))
            ]),
        textrecog=dict(
            visualizer=dict(
                type='TextRecogLocalVisualizer',
                name='visualizer',
                vis_backends=[dict(type='LocalVisBackend')]),
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk'),
                    ignore_empty=True,
                    min_size=2),
                dict(type='LoadOCRAnnotations', with_text=True),
                dict(
                    type='PackTextRecogInputs',
                    meta_keys=('img_path', 'ori_shape', 'img_shape',
                               'valid_ratio'))
            ]),
    )

    # Model config mode
    dataloader_name = f'{phase}_dataloader'
    if dataloader_name in cfg:
        dataset = cfg.get(dataloader_name).dataset
        visualizer = cfg.visualizer

        if mode == 'original':
            default_cfg = default_cfgs[infer_dataset_task(task, dataset)]
            dataset.pipeline = default_cfg['pipeline']

        return dataset, visualizer

    # Dataset config mode

    for key in cfg.keys():
        if key.endswith(phase) and cfg[key]['type'].endswith('Dataset'):
            dataset = cfg[key]
            default_cfg = default_cfgs[infer_dataset_task(
                task, dataset, key.lower())]
            visualizer = default_cfg['visualizer']
            dataset['pipeline'] = default_cfg['pipeline'] if dataset[
                'pipeline'] is None else dataset['pipeline']

            return dataset, visualizer

    raise ValueError(
        f'Unable to find "{phase}_dataloader" or any dataset variable ending '
        f'with "{phase}". Please check your config file or --phase argument '
        'and try again. More details can be found in the docstring of '
        'obtain_dataset_cfg function. Or, you may visit the documentation via '
        'https://mmocr.readthedocs.io/en/dev-1.x/user_guides/useful_tools.html#dataset-visualization-tool'  # noqa: E501
    )


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmocr'))

    dataset_cfg, visualizer_cfg = obtain_dataset_cfg(cfg, args.phase,
                                                     args.mode, args.task)
    dataset = DATASETS.build(dataset_cfg)
    visualizer = VISUALIZERS.build(visualizer_cfg)
    visualizer.dataset_meta = dataset.metainfo

    intermediate_imgs = []

    if dataset_cfg.type == 'ConcatDataset':
        for sub_dataset in dataset.datasets:
            sub_dataset.pipeline = InspectCompose(
                sub_dataset.pipeline.transforms, intermediate_imgs)
    else:
        dataset.pipeline = InspectCompose(dataset.pipeline.transforms,
                                          intermediate_imgs)

    # init visualization image number
    assert args.show_number > 0
    display_number = min(args.show_number, len(dataset))

    progress_bar = ProgressBar(display_number)
    # fetching items from dataset is a must for visualization
    for i, _ in zip(range(display_number), dataset):
        image_i = []
        result_i = [result['dataset_sample'] for result in intermediate_imgs]
        for k, datasample in enumerate(result_i):
            image = datasample.img
            image = image[..., [2, 1, 0]]  # bgr to rgb
            image_show = visualizer.add_datasample(
                'result',
                image,
                datasample,
                draw_pred=False,
                draw_gt=True,
                show=False)
            image_i.append(image_show)

        if args.mode == 'pipeline':
            image = make_grid([result for result in image_i],
                              [result['name'] for result in intermediate_imgs])
        else:
            image = image_i[-1]

        if hasattr(datasample, 'img_path'):
            filename = osp.basename(datasample.img_path)
        else:
            # some dataset have not image path
            filename = f'{i}.jpg'
        out_file = osp.join(args.output_dir,
                            filename) if args.output_dir is not None else None

        if out_file is not None:
            mmcv.imwrite(image[..., ::-1], out_file)

        if not args.not_show:
            visualizer.show(
                image, win_name=filename, wait_time=args.show_interval)

        intermediate_imgs.clear()
        progress_bar.update()


if __name__ == '__main__':
    main()
