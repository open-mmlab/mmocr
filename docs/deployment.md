# Deployment

We provide deployment tools under `tools/deployment` directory.

## Convert to ONNX (experimental)

We provide a script to convert the model to [ONNX](https://github.com/onnx/onnx) format. The converted model could be visualized by tools like [Netron](https://github.com/lutzroeder/netron). Besides, we also support comparing the output results between PyTorch and ONNX model.

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

| ARGS               | Type           | Description                                                                                        |
| ------------------ | -------------- | -------------------------------------------------------------------------------------------------- |
| `model_config`     | str            | The path to a model config file.                                                                   |
| `model_ckpt`       | str            | The path to a model checkpoint file.                                                               |
| `model_type`       | 'recog', 'det' | The model type of the config file.                                                                 |
| `image_path`       | str            | The path to input image file.                                                                      |
| `--output-file`    | str            | The path to output ONNX model. Defaults to `tmp.onnx`.                                             |
| `--device-id`      | int            | Which GPU to use. Defaults to 0.                                                                   |
| `--opset-version`  | int            | ONNX opset version. Defaults to 11.                                                                |
| `--verify`         | bool           | Determines whether to verify the correctness of an exported model. Defaults to `False`.            |
| `--verbose`        | bool           | Determines whether to print the architecture of the exported model. Defaults to `False`.           |
| `--show`           | bool           | Determines whether to visualize outputs of ONNXRuntime and PyTorch. Defaults to `False`.           |
| `--dynamic-export` | bool           | Determines whether to export ONNX model with dynamic input and output shapes. Defaults to `False`. |

:::{note}
This tool is still experimental. For now, some customized operators are not supported, and we only support a subset of detection and recognition algorithms.
:::

### List of supported models exportable to ONNX

The table below lists the models that are guaranteed to be exportable to ONNX and runnable in ONNX Runtime.

| Model  |                                                                      Config                                                                      | Dynamic Shape | Batch Inference |                  Note                  |
| :----: | :----------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :-------------: | :------------------------------------: |
| DBNet  |    [dbnet_r18_fpnc_1200e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py)    |       Y       |        N        |                                        |
| PSENet |     [psenet_r50_fpnf_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py)      |       Y       |        Y        |                                        |
| PSENet |   [psenet_r50_fpnf_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py)    |       Y       |        Y        |                                        |
| PANet  |   [panet_r18_fpem_ffm_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py)   |       Y       |        Y        |                                        |
| PANet  | [panet_r18_fpem_ffm_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |       Y       |        Y        |                                        |
|  CRNN  |            [crnn_academic_dataset.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/crnn_academic_dataset.py)             |       Y       |        Y        | CRNN only accepts input with height 32 |

:::{note}
- *All models above are tested with PyTorch==1.8.1 and onnxruntime-gpu == 1.8.1*
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmocr`.
:::

## Convert ONNX to TensorRT (experimental)

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

| ARGS               | Type           | Description                                                                                         |
| ------------------ | -------------- | --------------------------------------------------------------------------------------------------- |
| `model_config`     | str            | The path to a model config file.                                                                    |
| `model_type`       | 'recog', 'det' | The model type of the config file.                                                                  |
| `image_path`       | str            | The path to input image file.                                                                       |
| `onnx_file`        | str            | The path to input ONNX file.                                                                        |
| `--trt-file`       | str            | The path of output TensorRT model. Defaults to `tmp.trt`.                                           |
| `--max-shape`      | int * 4        | Maximum shape of model input.                                                                       |
| `--min-shape`      | int * 4        | Minimum shape of model input.                                                                       |
| `--workspace-size` | int            | Max workspace size in GiB. Defaults to 1.                                                           |
| `--fp16`           | bool           | Determines whether to export TensorRT with fp16 mode. Defaults to `False`.                          |
| `--verify`         | bool           | Determines whether to verify the correctness of an exported model. Defaults to `False`.             |
| `--show`           | bool           | Determines whether to show the output of ONNX and TensorRT. Defaults to `False`.                    |
| `--verbose`        | bool           | Determines whether to verbose logging messages while creating TensorRT engine. Defaults to `False`. |

:::{note}
This tool is still experimental. For now, some customized operators are not supported, and we only support a subset of detection and recognition algorithms.
:::

### List of supported models exportable to TensorRT

The table below lists the models that are guaranteed to be exportable to TensorRT engine and runnable in TensorRT.

| Model  |                                                                      Config                                                                      | Dynamic Shape | Batch Inference |                  Note                  |
| :----: | :----------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :-------------: | :------------------------------------: |
| DBNet  |    [dbnet_r18_fpnc_1200e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py)    |       Y       |        N        |                                        |
| PSENet |     [psenet_r50_fpnf_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py)      |       Y       |        Y        |                                        |
| PSENet |   [psenet_r50_fpnf_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py)    |       Y       |        Y        |                                        |
| PANet  |   [panet_r18_fpem_ffm_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py)   |       Y       |        Y        |                                        |
| PANet  | [panet_r18_fpem_ffm_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |       Y       |        Y        |                                        |
|  CRNN  |            [crnn_academic_dataset.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/crnn_academic_dataset.py)             |       Y       |        Y        | CRNN only accepts input with height 32 |

:::{note}
- *All models above are tested with PyTorch==1.8.1,  onnxruntime-gpu==1.8.1 and tensorrt==7.2.1.6*
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmocr`.
:::


## Evaluate ONNX and TensorRT Models (experimental)

We provide methods to evaluate TensorRT and ONNX models in `tools/deployment/deploy_test.py`.

### Prerequisite
To evaluate ONNX and TensorRT models, ONNX, ONNXRuntime and TensorRT should be installed first. Install `mmcv-full` with ONNXRuntime custom ops and TensorRT plugins follow [ONNXRuntime in mmcv](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) and [TensorRT plugin in mmcv](https://github.com/open-mmlab/mmcv/blob/master/docs/tensorrt_plugin.md).

### Usage

```bash
python tools/deploy_test.py \
    ${CONFIG_FILE} \
    ${MODEL_PATH} \
    ${MODEL_TYPE} \
    ${BACKEND} \
    --eval ${METRICS} \
    --device ${DEVICE}
```

### Description of all arguments

| ARGS           | Type                      | Description                                                                             |
| -------------- | ------------------------- | --------------------------------------------------------------------------------------- |
| `model_config` | str                       | The path to a model config file.                                                        |
| `model_file`   | str                       | The path to a TensorRT or an ONNX model file.                                           |
| `model_type`   | 'recog', 'det'            | Detection or recognition model to deploy.                                               |
| `backend`      | 'TensorRT', 'ONNXRuntime' | The backend for testing.                                                                |
| `--eval`       | 'acc', 'hmean-iou'        | The evaluation metrics. 'acc' for recognition models, 'hmean-iou' for detection models. |
| `--device`     | str                       | Device for evaluation. Defaults to `cuda:0`.                                            |

## Results and Models


<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8">Model</th>
    <th class="tg-9wq8">Config</th>
    <th class="tg-9wq8">Dataset</th>
    <th class="tg-9wq8">Metric</th>
    <th class="tg-9wq8">PyTorch</th>
    <th class="tg-9wq8">ONNX Runtime</th>
    <th class="tg-9wq8">TensorRT FP32</th>
    <th class="tg-9wq8">TensorRT FP16</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="3">DBNet</td>
    <td class="tg-9wq8" rowspan="3">dbnet_r18_fpnc_1200e_icdar2015.py<br></td>
    <td class="tg-9wq8" rowspan="3">icdar2015</td>
    <td class="tg-9wq8"><span style="font-style:normal">Recall</span><br></td>
    <td class="tg-9wq8">0.731</td>
    <td class="tg-9wq8">0.731</td>
    <td class="tg-9wq8">0.678</td>
    <td class="tg-9wq8">0.679</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision</td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.871</span></td>
    <td class="tg-9wq8">0.871</td>
    <td class="tg-9wq8">0.844</td>
    <td class="tg-9wq8">0.842</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><span style="font-style:normal">Hmean</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.795</span></td>
    <td class="tg-9wq8">0.795</td>
    <td class="tg-9wq8">0.752</td>
    <td class="tg-9wq8">0.752</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">DBNet*</td>
    <td class="tg-9wq8" rowspan="3">dbnet_r18_fpnc_1200e_icdar2015.py<br></td>
    <td class="tg-9wq8" rowspan="3">icdar2015</td>
    <td class="tg-9wq8"><span style="font-style:normal">Recall</span><br></td>
    <td class="tg-9wq8">0.720</td>
    <td class="tg-9wq8">0.720</td>
    <td class="tg-9wq8">0.720</td>
    <td class="tg-9wq8">0.718</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision</td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.868</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.868</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.868</span></td>
    <td class="tg-9wq8">0.868</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><span style="font-style:normal">Hmean</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.787</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.787</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.787</span></td>
    <td class="tg-9wq8">0.786</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">PSENet</td>
    <td class="tg-9wq8" rowspan="3">psenet_r50_fpnf_600e_icdar2015.py<br></td>
    <td class="tg-9wq8" rowspan="3">icdar2015</td>
    <td class="tg-9wq8"><span style="font-style:normal">Recall</span><br></td>
    <td class="tg-9wq8">0.753</td>
    <td class="tg-9wq8">0.753</td>
    <td class="tg-9wq8">0.753</td>
    <td class="tg-9wq8">0.752</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision</td>
    <td class="tg-9wq8">0.867</td>
    <td class="tg-9wq8">0.867</td>
    <td class="tg-9wq8">0.867</td>
    <td class="tg-9wq8">0.867</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><span style="font-style:normal">Hmean</span></td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.805</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">PANet</td>
    <td class="tg-9wq8" rowspan="3">panet_r18_fpem_ffm_600e_icdar2015.py<br></td>
    <td class="tg-9wq8" rowspan="3">icdar2015</td>
    <td class="tg-9wq8">Recall<br></td>
    <td class="tg-9wq8">0.740</td>
    <td class="tg-9wq8">0.740</td>
    <td class="tg-9wq8">0.687</td>
    <td class="tg-9wq8">N/A</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision</td>
    <td class="tg-9wq8">0.860</td>
    <td class="tg-9wq8">0.860</td>
    <td class="tg-9wq8">0.815</td>
    <td class="tg-9wq8">N/A</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Hmean</td>
    <td class="tg-9wq8">0.796</td>
    <td class="tg-9wq8">0.796</td>
    <td class="tg-9wq8">0.746</td>
    <td class="tg-9wq8">N/A</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="3">PANet*</td>
    <td class="tg-nrix" rowspan="3">panet_r18_fpem_ffm_600e_icdar2015.py<br></td>
    <td class="tg-nrix" rowspan="3">icdar2015</td>
    <td class="tg-nrix">Recall<br></td>
    <td class="tg-nrix">0.736</td>
    <td class="tg-nrix">0.736</td>
    <td class="tg-nrix">0.736</td>
    <td class="tg-nrix">N/A</td>
  </tr>
  <tr>
    <td class="tg-nrix">Precision</td>
    <td class="tg-nrix">0.857</td>
    <td class="tg-nrix">0.857</td>
    <td class="tg-nrix">0.857</td>
    <td class="tg-nrix">N/A</td>
  </tr>
  <tr>
    <td class="tg-nrix">Hmean</td>
    <td class="tg-nrix">0.792</td>
    <td class="tg-nrix">0.792</td>
    <td class="tg-nrix">0.792</td>
    <td class="tg-nrix">N/A</td>
  </tr>
  <tr>
    <td class="tg-9wq8">CRNN</td>
    <td class="tg-9wq8">crnn_academic_dataset.py<br></td>
    <td class="tg-9wq8">IIIT5K</td>
    <td class="tg-9wq8">Acc</td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.806</td>
  </tr>
</tbody>
</table>

:::{note}
- TensorRT upsampling operation is a little different from PyTorch. For DBNet and PANet, we suggest replacing upsampling operations with the nearest mode to operations with bilinear mode. [Here](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpem_ffm.py#L33) for PANet, [here](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpn_cat.py#L111) and [here](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpn_cat.py#L121) for DBNet. As is shown in the above table, networks with tag * mean the upsampling mode is changed.
- Note that changing upsampling mode reduces less performance compared with using the nearest mode. However, the weights of networks are trained through the nearest mode. To pursue the best performance, using bilinear mode for both training and TensorRT deployment is recommended.
- All ONNX and TensorRT models are evaluated with dynamic shapes on the datasets, and images are preprocessed according to the original config file.
- This tool is still experimental, and we only support a subset of detection and recognition algorithms for now.
:::
