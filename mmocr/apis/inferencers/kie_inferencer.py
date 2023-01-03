# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Dict, List, Optional, Sequence

import mmcv
import mmengine
import numpy as np
from mmengine.dataset import Compose
from mmengine.visualization import Visualizer

from mmocr.registry import DATASETS
from mmocr.structures import KIEDataSample
from mmocr.utils import ConfigType
from .base_mmocr_inferencer import BaseMMOCRInferencer, PredType

InputType = Dict
InputsType = Sequence[Dict]


class KIEInferencer(BaseMMOCRInferencer):
    """Key Information Extraction Inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "sdmgr_unet16_60e_wildreceipt" or
            "configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py".
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
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
            return Compose(pipeline_cfg)
        return super()._init_pipeline(cfg)

    def _init_visualizer(self, cfg: ConfigType) -> Optional[Visualizer]:
        """Initialize visualizers.

        Args:
            cfg (ConfigType): Config containing the visualizer information.

        Returns:
            Visualizer or None: Visualizer initialized with config.
        """
        visualizer = super()._init_visualizer(cfg)
        dataset = DATASETS.build(cfg.test_dataloader.dataset)
        visualizer.dataset_meta = dataset.metainfo
        return visualizer

    def _inputs_to_list(self, inputs: InputsType) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type.

        The inputs can be a dict or list[dict], where each dictionary contains
        following keys:

        - img (str or ndarray): Path to the image or the image itself. If KIE
          Inferencer is used in no-visual mode, this key is not required.
        - img_shape (tuple(int, int)): Image shape in (H, W). In
        - instances (list[dict]): A list of instances.
            - bbox (ndarray(dtype=np.float32)): Shape (4, ). Bounding box.
            - text (str): Annotation text.

        Each ``instance`` looks like the following:

        .. code-block:: python

        {
            # A nested list of 4 numbers representing the bounding box of
            # the instance, in (x1, y1, x2, y2) order.
            'bbox': np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                            dtype=np.int32),

            # List of texts.
            "texts": ['text1', 'text2', ...],
        }

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """

        processed_inputs = []

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        for single_input in inputs:
            if self.novisual:
                processed_input = copy.deepcopy(single_input)
                if 'img' not in single_input and \
                   'img_shape' not in single_input:
                    raise ValueError(
                        'KIEInferencer in no-visual mode '
                        'requires input has "img" or "img_shape", but both are'
                        ' not found.')
                if 'img' in single_input:
                    img = single_input['img']
                    if isinstance(img, str):
                        img_bytes = mmengine.fileio.get(img)
                        img = mmcv.imfrombytes(img_bytes)
                        processed_input['img'] = img
                    processed_input['img_shape'] = img.shape[:2]
                processed_inputs.append(processed_input)
            else:
                if 'img' not in single_input:
                    raise ValueError(
                        'This inferencer is constructed to '
                        'accept image inputs, but the input does not contain '
                        '"img" key.')
                if isinstance(single_input['img'], str):
                    processed_input = {
                        k: v
                        for k, v in single_input.items() if k != 'img'
                    }
                    processed_input['img_path'] = single_input['img']
                    processed_inputs.append(processed_input)
                elif isinstance(single_input['img'], np.ndarray):
                    processed_inputs.append(copy.deepcopy(single_input))
                else:
                    atype = type(single_input['img'])
                    raise ValueError(f'Unsupported input type: {atype}')
        return processed_inputs

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 return_vis: bool = False,
                 show: bool = False,
                 wait_time: int = 0,
                 draw_pred: bool = True,
                 pred_score_thr: float = 0.3,
                 img_out_dir: str = '',
                 print_result: bool = False,
                 pred_out_file: str = '',
                 **kwargs) -> dict:
        """Call the inferencer.

        The inputs for KIE Inferencer is special compared to other tasks.
        They can be a dict or list[dict], where each dictionary contains
        following keys:

        - img (str or ndarray): Path to the image or the image itself. If KIE
          Inferencer is used in no-visual mode, this key is not required.
        - img_shape (tuple(int, int)): Image shape in (H, W). In
        - instances (list[dict]): A list of instances.
            - bbox (ndarray(dtype=np.float32)): Shape (4, ). Bounding box.
            - text (str): Annotation text.

        Each ``instance`` looks like the following:

        .. code-block:: python

        {
            # A nested list of 4 numbers representing the bounding box of
            # the instance, in (x1, y1, x2, y2) order.
            'bbox': np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                            dtype=np.int32),

            # List of texts.
            "texts": ['text1', 'text2', ...],
        }

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Inference batch size. Defaults to 1.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_file: File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        return super().__call__(
            inputs,
            return_datasamples,
            batch_size,
            return_vis=return_vis,
            show=show,
            wait_time=wait_time,
            draw_pred=draw_pred,
            pred_score_thr=pred_score_thr,
            img_out_dir=img_out_dir,
            print_result=print_result,
            pred_out_file=pred_out_file,
            **kwargs)

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  img_out_dir: str = '') -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of images. Defaults to ''.
        """
        if self.visualizer is None or (not show and img_out_dir == ''
                                       and not return_vis):
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            img_num = str(self.num_visualized_imgs).zfill(8)
            assert 'img' in single_input or 'img_shape' in single_input
            if 'img' in single_input:
                if isinstance(single_input['img'], str):
                    img = mmcv.imread(single_input['img'])
                    img_name = osp.basename(single_input['img'])
                elif isinstance(single_input['img'], np.ndarray):
                    img = single_input['img'].copy()
                    img_name = f'{img_num}.jpg'
            elif 'img_shape' in single_input:
                img = np.zeros(single_input['img_shape'], dtype=np.uint8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Input does not contain either "img" or '
                                 '"img_shape"')

            out_file = osp.join(img_out_dir, img_name) if img_out_dir != '' \
                else None

            visualization = self.visualizer.add_datasample(
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
            results.append(visualization)
            self.num_visualized_imgs += 1

        return results

    def pred2dict(self, data_sample: KIEDataSample) -> Dict:
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
