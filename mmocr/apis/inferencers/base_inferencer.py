# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmengine import Config, InstanceData
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint

from mmocr.registry import MODELS, VISUALIZERS
from mmocr.utils import ConfigType

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]


class BaseInferencer:
    """Base inferencer.

    Args:
        model (str or ConfigType): Model config or the path to it.
        ckpt (str, optional): Path to the checkpoint.
        device (str, optional): Device to run inference. If None, the best
            device will be automatically used.
        show (bool): Whether to display the image in a popup window.
            Defaults to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        draw_pred (bool): Whether to draw predicted bounding boxes.
            Defaults to True.
        pred_score_thr (float): Minimum score of bboxes to draw.
            Defaults to 0.3.
        img_out_dir (str): Output directory of images. Defaults to ''.
        pred_out_file: File to save the inference results. If left as empty, no
            file will be saved.
        print_result (bool): Whether to print the result.
            Defaults to False.
    """

    func_kwargs = dict(
        preprocess=[],
        forward=[],
        visualize=[
            'show', 'wait_time', 'draw_pred', 'pred_score_thr', 'img_out_dir'
        ],
        postprocess=['print_result', 'pred_out_file'])
    func_order = dict(preprocess=0, forward=1, visualize=2, postprocess=3)

    def __init__(self,
                 config: Union[ConfigType, str],
                 ckpt: Optional[str],
                 device: Optional[str] = None,
                 **kwargs) -> None:
        # A global counter tracking the number of images processed, for
        # naming of the output images
        self.num_visualized_imgs = 0

        # Load config to cfg
        if isinstance(config, str):
            cfg = Config.fromfile(config)
        elif not isinstance(config, ConfigType):
            raise TypeError('config must be a filename or any ConfigType'
                            f'object, but got {type(cfg)}')
        if cfg.model.get('pretrained'):
            cfg.model.pretrained = None

        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self._init_model(cfg, ckpt, device)
        self._init_pipeline(cfg)
        self._init_visualizer(cfg)
        self.base_params = self.dispatch_kwargs(**kwargs)

    def _init_model(self, cfg: Union[ConfigType, str], ckpt: Optional[str],
                    device: str) -> None:
        """Initialize the model with the given config and checkpoint on the
        specific device."""
        model = MODELS.build(cfg.model)
        if ckpt is not None:
            ckpt = load_checkpoint(model, ckpt, map_location='cpu')
        model.cfg = cfg.model
        model.to(device)
        model.eval()
        self.model = model

    def _init_pipeline(self, cfg: ConfigType) -> None:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        # For inference, the key of ``instances`` is not used.
        pipeline_cfg[-1]['meta_keys'] = tuple(
            meta_key for meta_key in pipeline_cfg[-1]['meta_keys']
            if meta_key != 'instances')

        # Loading annotations is also not applicable
        idx = self._get_transform_idx(pipeline_cfg, 'LoadOCRAnnotations')
        if idx != -1:
            del pipeline_cfg[idx]

        self.file_pipeline = Compose(pipeline_cfg)

        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFile')
        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx] = dict(type='mmdet.LoadImageFromNDArray')
        self.ndarray_pipeline = Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1

    def _init_visualizer(self, cfg: ConfigType) -> None:
        """Initialize visualizers."""
        # TODO: We don't export images via backends since the interface
        # of the visualizer will have to be refactored.
        self.visualizer = None
        if 'visualizer' in cfg:
            self.visualizer = VISUALIZERS.build(cfg.visualizer)

    def dispatch_kwargs(self, **kwargs) -> Tuple[Dict, Dict, Dict, Dict]:
        """Dispatch kwargs to preprocess(), forward(), visualize() and
        postprocess() according to the actual demands."""
        results = [{}, {}, {}, {}]
        dispatched_kwargs = set()

        # Dispatch kwargs according to self.func_kwargs
        for func_name, func_kwargs in self.func_kwargs.items():
            for func_kwarg in func_kwargs:
                if func_kwarg in kwargs:
                    dispatched_kwargs.add(func_kwarg)
                    results[self.func_order[func_name]][func_kwarg] = kwargs[
                        func_kwarg]

        # Find if there is any kwargs that are not dispatched
        for kwarg in kwargs:
            if kwarg not in dispatched_kwargs:
                raise ValueError(f'Unknown kwarg: {kwarg}')

        return results

    def preprocess(self, inputs: InputsType) -> List[Dict]:
        """Process the inputs into a model-feedable format."""
        results = []
        for single_input in inputs:
            if isinstance(single_input, str):
                data_ = dict(img_path=single_input)
                results.append(self.file_pipeline(data_))
            elif isinstance(single_input, np.ndarray):
                data_ = dict(img=single_input)
                results.append(self.ndarray_pipeline(data_))
            else:
                raise ValueError(
                    f'Unsupported input type: {type(single_input)}')
        return results

    def __call__(self, user_inputs: InputsType,
                 **kwargs) -> Union[Dict, List[Dict]]:
        """Call the inferencer.

        Args:
            user_inputs: Inputs for the inferencer.
            kwargs: Keyword arguments for the inferencer.
        """

        # Detect if user_inputs are in a batch
        is_batch = isinstance(user_inputs, (list, tuple))
        inputs = user_inputs if is_batch else [user_inputs]

        params = self.dispatch_kwargs(**kwargs)
        preprocess_kwargs = self.base_params[0].copy()
        preprocess_kwargs.update(params[0])
        forward_kwargs = self.base_params[1].copy()
        forward_kwargs.update(params[1])
        visualize_kwargs = self.base_params[2].copy()
        visualize_kwargs.update(params[2])
        postprocess_kwargs = self.base_params[3].copy()
        postprocess_kwargs.update(params[3])

        data = self.preprocess(inputs, **preprocess_kwargs)
        preds = self.forward(data, **forward_kwargs)
        imgs = self.visualize(inputs, preds, **visualize_kwargs)
        results = self.postprocess(
            preds, imgs, is_batch=is_batch, **postprocess_kwargs)
        return results

    def forward(self, inputs: InputsType) -> PredType:
        with torch.no_grad():
            return self.model.test_step(inputs)

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  img_out_dir: str = '') -> ImgType:
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
        if not show and img_out_dir == '':
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img = mmcv.imread(single_input)
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

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

    def postprocess(self,
                    preds: PredType,
                    imgs: Optional[ImgType] = None,
                    is_batch: bool = False,
                    print_result: bool = False,
                    pred_out_file: str = '') -> Union[Dict, List[Dict]]:
        """Postprocess predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            imgs (Optional[np.ndarray]): Visualized predictions.
            is_batch (bool): Whether the inputs are in a batch.
                Defaults to False.
            print_result (bool): Whether to print the result.
                Defaults to False.
            pred_out_file (str): Output file name to store predictions
                without images. Supported file formats are “json”, “yaml/yml”
                and “pickle/pkl”. Defaults to ''.
        """

        results = []
        for i, pred in enumerate(preds):
            result = self._pred2dict(pred)
            results.append(result)
        if print_result:
            print(results)
        if pred_out_file != '':
            mmcv.dump(results, pred_out_file)
        if imgs is not None:
            for i, img in enumerate(imgs):
                results[i]['img'] = img
        if not is_batch:
            results = results[0]
        return results

    def _pred2dict(self, data_sample: InstanceData) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.
        """
        raise NotImplementedError
