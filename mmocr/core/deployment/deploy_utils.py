# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Any, Iterable

import numpy as np
import torch

from mmocr.models.textdet.detectors.single_stage_text_detector import \
    SingleStageTextDetector
from mmocr.models.textdet.detectors.text_detector_mixin import \
    TextDetectorMixin
from mmocr.models.textrecog.recognizer.encode_decode_recognizer import \
    EncodeDecodeRecognizer
from mmocr.registry import MODELS


def inference_with_session(sess, io_binding, input_name, output_names,
                           input_tensor):
    device_type = input_tensor.device.type
    device_id = input_tensor.device.index
    device_id = 0 if device_id is None else device_id
    io_binding.bind_input(
        name=input_name,
        device_type=device_type,
        device_id=device_id,
        element_type=np.float32,
        shape=input_tensor.shape,
        buffer_ptr=input_tensor.data_ptr())
    for name in output_names:
        io_binding.bind_output(name)
    sess.run_with_iobinding(io_binding)
    pred = io_binding.copy_outputs_to_cpu()
    return pred


@MODELS.register_module()
class ONNXRuntimeDetector(TextDetectorMixin, SingleStageTextDetector):
    """The class for evaluating onnx file of detection."""

    def __init__(self,
                 onnx_file: str,
                 cfg: Any,
                 device_id: int,
                 show_score: bool = False):
        if 'type' in cfg.model:
            cfg.model.pop('type')
        SingleStageTextDetector.__init__(self, **(cfg.model))
        TextDetectorMixin.__init__(self, show_score)
        import onnxruntime as ort

        # get the custom op path
        ort_custom_op_path = ''
        try:
            from mmcv.ops import get_onnxruntime_op_path
            ort_custom_op_path = get_onnxruntime_op_path()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with ONNXRuntime from source.')
        session_options = ort.SessionOptions()
        # register custom op for onnxruntime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = ort.InferenceSession(onnx_file, session_options)
        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})

        sess.set_providers(providers, options)

        self.sess = sess
        self.device_id = device_id
        self.io_binding = sess.io_binding()
        self.output_names = [_.name for _ in sess.get_outputs()]
        for name in self.output_names:
            self.io_binding.bind_output(name)
        self.cfg = cfg

    def forward_train(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def simple_test(self,
                    img: torch.Tensor,
                    img_metas: Iterable,
                    rescale: bool = False):
        onnx_pred = inference_with_session(self.sess, self.io_binding, 'input',
                                           self.output_names, img)
        onnx_pred = torch.from_numpy(onnx_pred[0])
        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(*(onnx_pred[i].unsqueeze(0)),
                                            [img_metas[i]], rescale)
                for i in range(len(img_metas))
            ]

        else:
            boundaries = [
                self.bbox_head.get_boundary(*onnx_pred, img_metas, rescale)
            ]

        return boundaries


@MODELS.register_module()
class ONNXRuntimeRecognizer(EncodeDecodeRecognizer):
    """The class for evaluating onnx file of recognition."""

    def __init__(self,
                 onnx_file: str,
                 cfg: Any,
                 device_id: int,
                 show_score: bool = False):
        if 'type' in cfg.model:
            cfg.model.pop('type')
        EncodeDecodeRecognizer.__init__(self, **(cfg.model))
        import onnxruntime as ort

        # get the custom op path
        ort_custom_op_path = ''
        try:
            from mmcv.ops import get_onnxruntime_op_path
            ort_custom_op_path = get_onnxruntime_op_path()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with ONNXRuntime from source.')
        session_options = ort.SessionOptions()
        # register custom op for onnxruntime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = ort.InferenceSession(onnx_file, session_options)
        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})

        sess.set_providers(providers, options)

        self.sess = sess
        self.device_id = device_id
        self.io_binding = sess.io_binding()
        self.output_names = [_.name for _ in sess.get_outputs()]
        for name in self.output_names:
            self.io_binding.bind_output(name)
        self.cfg = cfg

    def forward_train(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        if isinstance(imgs, list):
            for idx, each_img in enumerate(imgs):
                if each_img.dim() == 3:
                    imgs[idx] = each_img.unsqueeze(0)
            imgs = imgs[0]  # avoid aug_test
            img_metas = img_metas[0]
        else:
            if len(img_metas) == 1 and isinstance(img_metas[0], list):
                img_metas = img_metas[0]
        return self.simple_test(imgs, img_metas=img_metas)

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def simple_test(self,
                    img: torch.Tensor,
                    img_metas: Iterable,
                    rescale: bool = False):
        """Test function.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        onnx_pred = inference_with_session(self.sess, self.io_binding, 'input',
                                           self.output_names, img)
        onnx_pred = torch.from_numpy(onnx_pred[0])

        label_indexes, label_scores = self.label_convertor.tensor2idx(
            onnx_pred, img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results


@MODELS.register_module()
class TensorRTDetector(TextDetectorMixin, SingleStageTextDetector):
    """The class for evaluating TensorRT file of detection."""

    def __init__(self,
                 trt_file: str,
                 cfg: Any,
                 device_id: int,
                 show_score: bool = False):
        if 'type' in cfg.model:
            cfg.model.pop('type')
        SingleStageTextDetector.__init__(self, **(cfg.model))
        TextDetectorMixin.__init__(self, show_score)
        from mmcv.tensorrt import TRTWrapper, load_tensorrt_plugin
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with TensorRT from source.')
        model = TRTWrapper(
            trt_file, input_names=['input'], output_names=['output'])

        self.model = model
        self.device_id = device_id
        self.cfg = cfg

    def forward_train(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def simple_test(self,
                    img: torch.Tensor,
                    img_metas: Iterable,
                    rescale: bool = False):
        with torch.cuda.device(self.device_id), torch.no_grad():
            trt_pred = self.model({'input': img})['output']
        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(*(trt_pred[i].unsqueeze(0)),
                                            [img_metas[i]], rescale)
                for i in range(len(img_metas))
            ]

        else:
            boundaries = [
                self.bbox_head.get_boundary(*trt_pred, img_metas, rescale)
            ]

        return boundaries


@MODELS.register_module()
class TensorRTRecognizer(EncodeDecodeRecognizer):
    """The class for evaluating TensorRT file of recognition."""

    def __init__(self,
                 trt_file: str,
                 cfg: Any,
                 device_id: int,
                 show_score: bool = False):
        if 'type' in cfg.model:
            cfg.model.pop('type')
        EncodeDecodeRecognizer.__init__(self, **(cfg.model))
        from mmcv.tensorrt import TRTWrapper, load_tensorrt_plugin
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with TensorRT from source.')
        model = TRTWrapper(
            trt_file, input_names=['input'], output_names=['output'])

        self.model = model
        self.device_id = device_id
        self.cfg = cfg

    def forward_train(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        if isinstance(imgs, list):
            for idx, each_img in enumerate(imgs):
                if each_img.dim() == 3:
                    imgs[idx] = each_img.unsqueeze(0)
            imgs = imgs[0]  # avoid aug_test
            img_metas = img_metas[0]
        else:
            if len(img_metas) == 1 and isinstance(img_metas[0], list):
                img_metas = img_metas[0]
        return self.simple_test(imgs, img_metas=img_metas)

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def simple_test(self,
                    img: torch.Tensor,
                    img_metas: Iterable,
                    rescale: bool = False):
        """Test function.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        with torch.cuda.device(self.device_id), torch.no_grad():
            trt_pred = self.model({'input': img})['output']

        label_indexes, label_scores = self.label_convertor.tensor2idx(
            trt_pred, img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results
