import tempfile
from functools import partial

import mmcv
import numpy as np
import pytest
import torch
from mmdet.models import build_detector
from packaging import version

from mmocr.core.deployment import (ONNXRuntimeDetector, ONNXRuntimeRecognizer,
                                   TensorRTDetector, TensorRTRecognizer)


@pytest.mark.skipif(torch.__version__ == 'parrots', reason='skip parrots.')
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('1.4.0'),
    reason='skip if torch=1.3.x')
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='skip if on cpu device')
def test_detector_wrapper():
    try:
        import onnxruntime as ort  # noqa: F401
        import tensorrt as trt
        from mmcv.tensorrt import (onnx2trt, save_trt_engine)
    except ImportError:
        pytest.skip('ONNXRuntime or TensorRT is not available.')

    cfg = dict(
        model=dict(
            type='DBNet',
            backbone=dict(
                type='ResNet',
                depth=18,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                norm_cfg=dict(type='BN', requires_grad=True),
                init_cfg=dict(
                    type='Pretrained', checkpoint='torchvision://resnet18'),
                norm_eval=False,
                style='caffe'),
            neck=dict(
                type='FPNC',
                in_channels=[64, 128, 256, 512],
                lateral_channels=256),
            bbox_head=dict(
                type='DBHead',
                text_repr_type='quad',
                in_channels=256,
                loss=dict(type='DBLoss', alpha=5.0, beta=10.0,
                          bbce_loss=True)),
            train_cfg=None,
            test_cfg=None))

    cfg = mmcv.Config(cfg)

    pytorch_model = build_detector(cfg.model, None, None)

    # prepare data
    inputs = torch.rand(1, 3, 224, 224)
    img_metas = [{
        'img_shape': [1, 3, 224, 224],
        'ori_shape': [1, 3, 224, 224],
        'pad_shape': [1, 3, 224, 224],
        'filename': None,
        'scale_factor': np.array([1, 1, 1, 1])
    }]

    pytorch_model.forward = pytorch_model.forward_dummy
    with tempfile.TemporaryDirectory() as tmpdirname:
        onnx_path = f'{tmpdirname}/tmp.onnx'
        with torch.no_grad():
            torch.onnx.export(
                pytorch_model,
                inputs,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                export_params=True,
                keep_initializers_as_inputs=False,
                verbose=False,
                opset_version=11)

        # TensorRT part
        def get_GiB(x: int):
            """return x GiB."""
            return x * (1 << 30)

        trt_path = onnx_path.replace('.onnx', '.trt')
        min_shape = [1, 3, 224, 224]
        max_shape = [1, 3, 224, 224]
        # create trt engine and wrapper
        opt_shape_dict = {'input': [min_shape, min_shape, max_shape]}
        max_workspace_size = get_GiB(1)
        trt_engine = onnx2trt(
            onnx_path,
            opt_shape_dict,
            log_level=trt.Logger.ERROR,
            fp16_mode=False,
            max_workspace_size=max_workspace_size)
        save_trt_engine(trt_engine, trt_path)
        print(f'Successfully created TensorRT engine: {trt_path}')

        wrap_onnx = ONNXRuntimeDetector(onnx_path, cfg, 0)
        wrap_trt = TensorRTDetector(trt_path, cfg, 0)

    assert isinstance(wrap_onnx, ONNXRuntimeDetector)
    assert isinstance(wrap_trt, TensorRTDetector)

    with torch.no_grad():
        onnx_outputs = wrap_onnx.simple_test(inputs, img_metas, rescale=False)
        trt_outputs = wrap_onnx.simple_test(inputs, img_metas, rescale=False)

    assert isinstance(onnx_outputs[0], dict)
    assert isinstance(trt_outputs[0], dict)
    assert 'boundary_result' in onnx_outputs[0]
    assert 'boundary_result' in trt_outputs[0]


@pytest.mark.skipif(torch.__version__ == 'parrots', reason='skip parrots.')
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('1.4.0'),
    reason='skip if torch=1.3.x')
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='skip if on cpu device')
def test_recognizer_wrapper():
    try:
        import onnxruntime as ort  # noqa: F401
        import tensorrt as trt
        from mmcv.tensorrt import (onnx2trt, save_trt_engine)
    except ImportError:
        pytest.skip('ONNXRuntime or TensorRT is not available.')

    cfg = dict(
        label_convertor=dict(
            type='CTCConvertor',
            dict_type='DICT36',
            with_unknown=False,
            lower=True),
        model=dict(
            type='CRNNNet',
            preprocessor=None,
            backbone=dict(
                type='VeryDeepVgg', leaky_relu=False, input_channels=1),
            encoder=None,
            decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
            loss=dict(type='CTCLoss'),
            label_convertor=dict(
                type='CTCConvertor',
                dict_type='DICT36',
                with_unknown=False,
                lower=True),
            pretrained=None),
        train_cfg=None,
        test_cfg=None)

    cfg = mmcv.Config(cfg)

    pytorch_model = build_detector(cfg.model, None, None)

    # prepare data
    inputs = torch.rand(1, 1, 32, 32)
    img_metas = [{
        'img_shape': [1, 1, 32, 32],
        'ori_shape': [1, 1, 32, 32],
        'pad_shape': [1, 1, 32, 32],
        'filename': None,
        'scale_factor': np.array([1, 1, 1, 1])
    }]

    pytorch_model.forward = partial(
        pytorch_model.forward,
        img_metas=img_metas,
        return_loss=False,
        rescale=True)
    with tempfile.TemporaryDirectory() as tmpdirname:
        onnx_path = f'{tmpdirname}/tmp.onnx'
        with torch.no_grad():
            torch.onnx.export(
                pytorch_model,
                inputs,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                export_params=True,
                keep_initializers_as_inputs=False,
                verbose=False,
                opset_version=11)

        # TensorRT part
        def get_GiB(x: int):
            """return x GiB."""
            return x * (1 << 30)

        trt_path = onnx_path.replace('.onnx', '.trt')
        min_shape = [1, 1, 32, 32]
        max_shape = [1, 1, 32, 32]
        # create trt engine and wrapper
        opt_shape_dict = {'input': [min_shape, min_shape, max_shape]}
        max_workspace_size = get_GiB(1)
        trt_engine = onnx2trt(
            onnx_path,
            opt_shape_dict,
            log_level=trt.Logger.ERROR,
            fp16_mode=False,
            max_workspace_size=max_workspace_size)
        save_trt_engine(trt_engine, trt_path)
        print(f'Successfully created TensorRT engine: {trt_path}')

        wrap_onnx = ONNXRuntimeRecognizer(onnx_path, cfg, 0)
        wrap_trt = TensorRTRecognizer(trt_path, cfg, 0)

    assert isinstance(wrap_onnx, ONNXRuntimeRecognizer)
    assert isinstance(wrap_trt, TensorRTRecognizer)

    with torch.no_grad():
        onnx_outputs = wrap_onnx.simple_test(inputs, img_metas, rescale=False)
        trt_outputs = wrap_onnx.simple_test(inputs, img_metas, rescale=False)

    assert isinstance(onnx_outputs[0], dict)
    assert isinstance(trt_outputs[0], dict)
    assert 'text' in onnx_outputs[0]
    assert 'text' in trt_outputs[0]
