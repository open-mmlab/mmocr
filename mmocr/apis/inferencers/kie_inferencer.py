# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Any, Dict, List, Optional, Sequence, Union

import mmcv
import mmengine
import numpy as np
from mmengine.dataset import Compose, pseudo_collate
from mmengine.runner.checkpoint import _load_checkpoint

from mmocr.registry import DATASETS
from mmocr.structures import KIEDataSample
from mmocr.utils import ConfigType
from .base_mmocr_inferencer import BaseMMOCRInferencer, ModelType, PredType

InputType = Dict
InputsType = Sequence[Dict]


class KIEInferencer(BaseMMOCRInferencer):
    """Key Information Extraction Inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "sdmgr_unet16_60e_wildreceipt" or
            "configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py".
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to "mmocr".
    """

    def __init__(self,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmocr') -> None:
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)
        self._load_metainfo_to_visualizer(weights, self.cfg)
        self.collate_fn = self.kie_collate

    def _load_metainfo_to_visualizer(self, weights: Optional[str],
                                     cfg: ConfigType) -> None:
        """Load meta information to visualizer."""
        if hasattr(self, 'visualizer'):
            if weights is not None:
                w = _load_checkpoint(weights, map_location='cpu')
                if w and 'meta' in w and 'dataset_meta' in w['meta']:
                    self.visualizer.dataset_meta = w['meta']['dataset_meta']
                    return
            if 'test_dataloader' in cfg:
                dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
                dataset_cfg['lazy_init'] = True
                dataset_cfg['metainfo'] = None
                dataset = DATASETS.build(dataset_cfg)
                self.visualizer.dataset_meta = dataset.metainfo
            else:
                raise ValueError(
                    'KIEVisualizer requires meta information from weights or '
                    'test dataset, but none of them is provided.')

    def _init_pipeline(self, cfg: ConfigType) -> None:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        idx = self._get_transform_idx(pipeline_cfg, 'LoadKIEAnnotations')
        if idx == -1:
            raise ValueError(
                'LoadKIEAnnotations is not found in the test pipeline')
        pipeline_cfg[idx]['with_label'] = False
        self.novisual = all(
            self._get_transform_idx(pipeline_cfg, t) == -1
            for t in self.loading_transforms)
        # Remove Resize from test_pipeline, since SDMGR requires bbox
        # annotations to be resized together with pictures, but visualization
        # loads the original image from the disk.
        # TODO: find a more elegant way to fix this
        idx = self._get_transform_idx(pipeline_cfg, 'Resize')
        if idx != -1:
            pipeline_cfg.pop(idx)
        # If it's in non-visual mode, self.pipeline will be specified.
        # Otherwise, file_pipeline and ndarray_pipeline will be specified.
        if self.novisual:
            return Compose(pipeline_cfg)
        return super()._init_pipeline(cfg)

    @staticmethod
    def kie_collate(data_batch: Sequence) -> Any:
        """A collate function designed for KIE, where the first element (input)
        is a dict and we only want to keep it as-is instead of batching
        elements inside.

        Returns:
            Any: Transversed Data in the same format as the data_itement of
            ``data_batch``.
        """  # noqa: E501
        transposed = list(zip(*data_batch))
        for i in range(1, len(transposed)):
            transposed[i] = pseudo_collate(transposed[i])
        return transposed

    def _inputs_to_list(self, inputs: InputsType) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type.

        The inputs can be a dict or list[dict], where each dictionary contains
        following keys:

        - img (str or ndarray): Path to the image or the image itself. If KIE
          Inferencer is used in no-visual mode, this key is not required.
          Note: If it's an numpy array, it should be in BGR order.
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

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  save_vis: bool = False,
                  img_out_dir: str = '') -> Union[List[np.ndarray], None]:
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
            save_vis (bool): Whether to save the visualization result. Defaults
                to False.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if self.visualizer is None or not (show or save_vis or return_vis):
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            assert 'img' in single_input or 'img_shape' in single_input
            if 'img' in single_input:
                if isinstance(single_input['img'], str):
                    img_bytes = mmengine.fileio.get(single_input['img'])
                    img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                elif isinstance(single_input['img'], np.ndarray):
                    img = single_input['img'].copy()[:, :, ::-1]  # To RGB
            elif 'img_shape' in single_input:
                img = np.zeros(single_input['img_shape'], dtype=np.uint8)
            else:
                raise ValueError('Input does not contain either "img" or '
                                 '"img_shape"')
            img_name = osp.splitext(osp.basename(pred.img_path))[0]

            if save_vis and img_out_dir:
                out_file = osp.splitext(img_name)[0]
                out_file = f'{out_file}.jpg'
                out_file = osp.join(img_out_dir, out_file)
            else:
                out_file = None

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
