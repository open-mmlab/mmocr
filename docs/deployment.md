## Deployment

We provide deployment tools under `tools/deployment` directory.

### Convert to ONNX (experimental)

We provide a script to convert model to [ONNX](https://github.com/onnx/onnx) format. The converted model could be visualized by tools like [Netron](https://github.com/lutzroeder/netron). Besides, we also support comparing the output results between Pytorch and ONNX model.

```bash
python tools/deployment/pytorch2onnx.py
    ${MODEL_CONFIG_PATH} \
    ${MODEL_CKPT_PATH} \
    ${MODEL_TYPE} \
    ${IMAGE_PATH} \
    --output-file ${OUTPUT_FILE} \
    --device-id ${DEVICE_ID} \
    --opset-version ${OPSET_VERSION} \
    --verify \
    --verbose \
    --show \
    --dynamic-export
```

Description of arguments:

- `model_config` : The path of a model config file.
- `model_ckpt` : The path of a model checkpoint file.
- `model_type` : The model type of the config file, options: `recog`, `det`.
- `image_path` : The path to input image file.
- `--output-file`: The path of output ONNX model. If not specified, it will be set to `tmp.onnx`.
- `--device-id`: Which gpu to use. If not specified, it will be set to 0.
- `--opset-version` : ONNX opset version, default to 11.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.
- `--verbose`: Determines whether to print the architecture of the exported model. If not specified, it will be set to `False`.
- `--show`: Determines whether to visualize outputs of ONNXRuntime and pytorch. If not specified, it will be set to `False`.
- `--dynamic-export`: Determines whether to export ONNX model with dynamic input and output shapes. If not specified, it will be set to `False`.

**Note**: This tool is still experimental. Some customized operators are not supported for now. And we only support `detection` and `recognition` for now.

#### List of supported models exportable to ONNX

The table below lists the models that are guaranteed to be exportable to ONNX and runnable in ONNX Runtime.

|  Model |                                                                      Config                                                                      | Dynamic Shape | Batch Inference | Note |
|:------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|:---------------:|:----:|
|  DBNet |    [dbnet_r18_fpnc_1200e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py)    |       Y       |        N        |      |
| PSENet |      [psenet_r50_fpnf_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py)     |       Y       |        Y        |      |
| PSENet |    [psenet_r50_fpnf_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py)   |       Y       |        Y        |      |
|  PANet |   [panet_r18_fpem_ffm_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py)   |       Y       |        Y        |      |
|  PANet | [panet_r18_fpem_ffm_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |       Y       |        Y        |      |
|  CRNN  |             [crnn_academic_dataset.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/crnn_academic_dataset.py)            |       Y       |        Y        |      |

**Notes**:

- *All models above are tested with Pytorch==1.8.1 and onnxruntime==1.7.0*
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, please try to solve them by yourself.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmocr`.

### Convert ONNX to TensorRT (experimental)

We also provide a script to convert [ONNX](https://github.com/onnx/onnx) model to [TensorRT](https://github.com/NVIDIA/TensorRT) format. Besides, we support comparing the output results between ONNX and TensorRT model.


```bash
python tools/deployment/onnx2tensorrt.py
    ${MODEL_CONFIG_PATH} \
    ${MODEL_TYPE} \
    ${IMAGE_PATH} \
    ${ONNX_FILE} \
    --trt-file ${OUT_TENSORRT} \
    --max-shape INT INT INT INT \
    --min-shape INT INT INT INT \
    --workspace-size INT \
    --fp16 \
    --verify \
    --show \
    --verbose
```

Description of arguments:

- `model_config` : The path of a model config file.
- `model_type` :The model type of the config file, options:
- `image_path` : The path to input image file.
- `onnx_file` : The path to input ONNX file.
- `--trt-file` : The path of output TensorRT model. If not specified, it will be set to `tmp.trt`.
- `--max-shape` : Maximum shape of model input.
- `--min-shape` : Minimum shape of model input.
- `--workspace-size`: Max workspace size in GiB. If not specified, it will be set to 1 GiB.
- `--fp16`: Determines whether to export TensorRT with fp16 mode. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.
- `--show`: Determines whether to show the output of ONNX and TensorRT. If not specified, it will be set to `False`.
- `--verbose`: Determines whether to verbose logging messages while creating TensorRT engine. If not specified, it will be set to `False`.

**Note**: This tool is still experimental. Some customized operators are not supported for now. We only support `detection` and `recognition` for now.

#### List of supported models exportable to TensorRT

The table below lists the models that are guaranteed to be exportable to TensorRT engine and runnable in TensorRT.

|  Model |                                                                      Config                                                                      | Dynamic Shape | Batch Inference | Note |
|:------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|:---------------:|:----:|
|  DBNet |    [dbnet_r18_fpnc_1200e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py)    |       Y       |        N        |      |
| PSENet |      [psenet_r50_fpnf_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py)     |       Y       |        Y        |      |
| PSENet |    [psenet_r50_fpnf_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py)   |       Y       |        Y        |      |
|  PANet |   [panet_r18_fpem_ffm_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py)   |       Y       |        Y        |      |
|  PANet | [panet_r18_fpem_ffm_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |       Y       |        Y        |      |
|  CRNN  |             [crnn_academic_dataset.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/crnn_academic_dataset.py)            |       Y       |        Y        | CRNN only accepts input with height 32 |

**Notes**:

- *All models above are tested with Pytorch==1.8.1,  onnxruntime==1.7.0 and tensorrt==7.2.1.6*
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, please try to solve them by yourself.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmocr`.
