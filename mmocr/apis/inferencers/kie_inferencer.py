# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Sequence

import mmcv
import numpy as np
from mmengine.dataset import Compose

from mmocr.registry import DATASETS, VISUALIZERS
from mmocr.structures import KIEDataSample
from mmocr.utils import ConfigType
from .base_mmocr_inferencer import BaseMMOCRInferencer, PredType

InputType = Dict
InputsType = Sequence[Dict]


class KIEInferencer(BaseMMOCRInferencer):
    """
    Inputs:
        dict or list[dict]: A dictionary containing the following keys:
            'bbox', 'texts', ` in this format:

            - img (str or ndarray): Path to the image or the image itself.
            - img_shape (tuple(int, int)): Image shape in (H, W). In
            - instances (list[dict]): A list of instances.
              - bbox (ndarray(dtype=np.float32)): Shape (4, ). Bounding box.
              - text (str): Annotation text.

            .. code-block:: python

            {
                # A nested list of 4 numbers representing the bounding box of
                # the instance, in (x1, y1, x2, y2) order.
                'bbox': np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                                dtype=np.int32),

                # List of texts.
                "texts": ['text1', 'text2', ...],
            }
    """

    def _init_pipeline(self, cfg: ConfigType) -> None:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        idx = self._get_transform_idx(pipeline_cfg, 'LoadKIEAnnotations')
        if idx == -1:
            raise ValueError(
                'LoadKIEAnnotations is not found in the test pipeline')
        pipeline_cfg[idx]['with_label'] = False
        self.novisual = self._get_transform_idx(pipeline_cfg,
                                                'LoadImageFromFile') == -1
        # If it's in non-visual mode, self.pipeline will be specified.
        # Otherwise, file_pipeline and ndarray_pipeline will be specified.
        if self.novisual:
            self.pipeline = Compose(pipeline_cfg)
        else:
            return super()._init_pipeline(cfg)

    def _init_visualizer(self, cfg: ConfigType) -> None:
        """Initialize visualizers."""
        # TODO: We don't export images via backends since the interface
        # of the visualizer will have to be refactored.
        self.visualizer = None
        if 'visualizer' in cfg:
            self.visualizer = VISUALIZERS.build(cfg.visualizer)
            dataset = DATASETS.build(cfg.test_dataloader.dataset)
            self.visualizer.dataset_meta = dataset.metainfo

    def preprocess(self, inputs: InputsType) -> List[Dict]:
        results = []
        for single_input in inputs:
            if self.novisual:
                if 'img' not in single_input and \
                  'img_shape' not in single_input:
                    raise ValueError(
                        'KIEInferencer in no-visual mode '
                        'requires input has "img" or "img_shape", but both are'
                        ' not found.')
                if 'img' in single_input:
                    new_input = {
                        k: v
                        for k, v in single_input.items() if k != 'img'
                    }
                    img = single_input['img']
                    if isinstance(img, str):
                        img = mmcv.imread(img)
                    new_input['img_shape'] = img.shape[::2]
                results.append(self.pipeline(new_input))
            else:
                if 'img' not in single_input:
                    raise ValueError(
                        'This inferencer is constructed to '
                        'accept image inputs, but the input does not contain '
                        '"img" key.')
                if isinstance(single_input['img'], str):
                    data_ = {
                        k: v
                        for k, v in single_input.items() if k != 'img'
                    }
                    data_['img_path'] = single_input['img']
                    results.append(self.file_pipeline(data_))
                elif isinstance(single_input['img'], np.ndarray):
                    results.append(self.ndarray_pipeline(single_input))
                else:
                    atype = type(single_input['img'])
                    raise ValueError(f'Unsupported input type: {atype}')
        return self._collate(results)

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  img_out_dir: str = '') -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of images. Defaults to ''.
        """
        if self.visualizer is None or not show and img_out_dir == '':
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            assert 'img' in single_input or 'img_shape' in single_input
            if 'img' in single_input:
                if isinstance(single_input['img'], str):
                    img = mmcv.imread(single_input['img'])
                    img_name = osp.basename(single_input['img'])
                elif isinstance(single_input['img'], np.ndarray):
                    img = single_input['img'].copy()
                    img_num = str(self.num_visualized_imgs).zfill(8)
                    img_name = f'{img_num}.jpg'
            elif 'img_shape' in single_input:
                img = np.zeros(single_input['img_shape'], dtype=np.uint8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Input does not contain either "img" or '
                                 '"img_shape"')

            out_file = osp.join(img_out_dir, img_name) if img_out_dir != '' \
                else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
            )
            results.append(img)
            self.num_visualized_imgs += 1

        return results

    def _pred2dict(self, data_sample: KIEDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (TextRecogDataSample): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        pred = data_sample.pred_instances
        result['scores'] = pred.scores.cpu().numpy().tolist()
        result['edge_scores'] = pred.edge_scores.cpu().numpy().tolist()
        result['edge_labels'] = pred.edge_labels.cpu().numpy().tolist()
        result['labels'] = pred.labels.cpu().numpy().tolist()
        return result
