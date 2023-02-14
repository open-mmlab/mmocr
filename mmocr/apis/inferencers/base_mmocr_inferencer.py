# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mmcv
import mmengine
import numpy as np
from mmengine.dataset import Compose
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.structures import InstanceData
from rich.progress import track
from torch import Tensor

from mmocr.utils import ConfigType, register_all_modules

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


class BaseMMOCRInferencer(BaseInferencer):
    """Base inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "dbnet_resnet18_fpnc_1200e_icdar2015" or
            "configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py".
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

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'draw_pred', 'pred_score_thr',
        'save_vis'
    }
    postprocess_kwargs: set = {
        'print_result', 'return_datasample', 'save_pred'
    }
    loading_transforms: list = ['LoadImageFromFile', 'LoadImageFromNDArray']

    def __init__(self,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmocr') -> None:
        # A global counter tracking the number of images given in the form
        # of ndarray, for naming the output images
        self.num_unnamed_imgs = 0
        register_all_modules()
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(inputs, batch_size)
        yield from map(self.collate_fn, chunked_data)

    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from inputs.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    inputs_ = next(inputs_iter)
                    pipe_out = self.pipeline(inputs_)
                    if pipe_out['data_samples'].get('img_path') is None:
                        pipe_out['data_samples'].set_metainfo(
                            dict(img_path=f'{self.num_unnamed_imgs}.jpg'))
                        self.num_unnamed_imgs += 1
                    chunk_data.append((inputs_, pipe_out))
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 progress_bar: bool = True,
                 return_vis: bool = False,
                 show: bool = False,
                 wait_time: int = 0,
                 draw_pred: bool = True,
                 pred_score_thr: float = 0.3,
                 out_dir: str = 'results/',
                 save_vis: bool = False,
                 save_pred: bool = False,
                 print_result: bool = False,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer. It can be a path
                to image / image directory, or an array, or a list of these.
                Note: If it's an numpy array, it should be in BGR order.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Inference batch size. Defaults to 1.
            progress_bar (bool): Whether to show a progress bar. Defaults to
                True.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            out_dir (str): Output directory of results. Defaults to 'results/'.
            save_vis (bool): Whether to save the visualization results to
                "out_dir". Defaults to False.
            save_pred (bool): Whether to save the inference results to
                "out_dir". Defaults to False.
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
            dict: Inference and visualization results, mapped from
                "predictions" and "visualization".
        """
        if (save_vis or save_pred) and not out_dir:
            raise ValueError('out_dir must be specified when save_vis or '
                             'save_pred is True!')
        if out_dir:
            img_out_dir = osp.join(out_dir, 'vis')
            pred_out_dir = osp.join(out_dir, 'preds')
        else:
            img_out_dir, pred_out_dir = '', ''
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(
            return_vis=return_vis,
            show=show,
            wait_time=wait_time,
            draw_pred=draw_pred,
            pred_score_thr=pred_score_thr,
            save_vis=save_vis,
            save_pred=save_pred,
            print_result=print_result,
            **kwargs)

        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)
        results = {'predictions': [], 'visualization': []}
        for ori_inputs, data in track(
                inputs, description='Inference', disable=not progress_bar):
            preds = self.forward(data, **forward_kwargs)
            visualization = self.visualize(
                ori_inputs, preds, img_out_dir=img_out_dir, **visualize_kwargs)
            batch_res = self.postprocess(
                preds,
                visualization,
                return_datasamples,
                pred_out_dir=pred_out_dir,
                **postprocess_kwargs)
            results['predictions'].extend(batch_res['predictions'])
            if batch_res['visualization'] is not None:
                results['visualization'].extend(batch_res['visualization'])
        return results

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        # For inference, the key of ``instances`` is not used.
        if 'meta_keys' in pipeline_cfg[-1]:
            pipeline_cfg[-1]['meta_keys'] = tuple(
                meta_key for meta_key in pipeline_cfg[-1]['meta_keys']
                if meta_key != 'instances')

        # Loading annotations is also not applicable
        idx = self._get_transform_idx(pipeline_cfg, 'LoadOCRAnnotations')
        if idx != -1:
            del pipeline_cfg[idx]

        for transform in self.loading_transforms:
            load_img_idx = self._get_transform_idx(pipeline_cfg, transform)
            if load_img_idx != -1:
                pipeline_cfg[load_img_idx]['type'] = 'InferencerLoader'
                break
        if load_img_idx == -1:
            raise ValueError(
                f'None of {self.loading_transforms} is found in the test '
                'pipeline')

        return Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1

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
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()[:, :, ::-1]  # to RGB
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')
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

    def postprocess(
        self,
        preds: PredType,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasample: bool = False,
        print_result: bool = False,
        save_pred: bool = False,
        pred_out_dir: str = '',
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            return_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            save_pred (bool): Whether to save the inference result. Defaults to
                False.
            pred_out_dir: File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.

            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
                :meth:`forward` and processed in :meth:`postprocess`.
                If ``return_datasample=False``, it usually should be a
                json-serializable dict containing only basic data elements such
                as strings and numbers.
        """
        result_dict = {}
        results = preds
        if not return_datasample:
            results = []
            for pred in preds:
                result = self.pred2dict(pred)
                if save_pred and pred_out_dir:
                    pred_name = osp.splitext(osp.basename(pred.img_path))[0]
                    pred_name = f'{pred_name}.json'
                    pred_out_file = osp.join(pred_out_dir, pred_name)
                    mmengine.dump(result, pred_out_file)
                results.append(result)
        # Add img to the results after printing and dumping
        result_dict['predictions'] = results
        if print_result:
            print(result_dict)
        result_dict['visualization'] = visualization
        return result_dict

    def pred2dict(self, data_sample: InstanceData) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.
        """
        raise NotImplementedError

    def _array2list(self, array: Union[Tensor, np.ndarray,
                                       List]) -> List[float]:
        """Convert a tensor or numpy array to a list.

        Args:
            array (Union[Tensor, np.ndarray]): The array to be converted.

        Returns:
            List[float]: The converted list.
        """
        if isinstance(array, Tensor):
            return array.detach().cpu().numpy().tolist()
        if isinstance(array, np.ndarray):
            return array.tolist()
        if isinstance(array, list):
            array = [self._array2list(arr) for arr in array]
        return array
