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
|  CRNN  |             [crnn_academic_dataset.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/crnn_academic_dataset.py)            |       Y       |        Y        | CRNN only accepts input with height 32 |

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


### Evaluate ONNX and TensorRT Models (experimental)

We provide methods to evaluate TensorRT and ONNX models in `tools/deployment/deploy_test.py`.

#### Prerequisite
To evaluate ONNX and TensorRT models, onnx, onnxruntime and TensorRT should be installed first. Install `mmcv-full` with ONNXRuntime custom ops and TensorRT plugins follow [ONNXRuntime in mmcv](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) and [TensorRT plugin in mmcv](https://github.com/open-mmlab/mmcv/blob/master/docs/tensorrt_plugin.md).

#### Usage

```bash
python tools/deploy_test.py \
    ${CONFIG_FILE} \
    ${MODEL_PATH} \
    ${MODEL_TYPE} \
    ${BACKEND} \
    --eval ${METRICS} \
    --device ${DEVICE}
```

#### Description of all arguments

- `model_config`: The path of a model config file.
- `model_file`: The path of a TensorRT or an ONNX model file.
- `model_type`: Detection or recognition model to deploy. Choose `recog` or `det`.
- `backend`: The backend for testing, choose TensorRT or ONNXRuntime.
- `--eval`: The evaluation metrics. `acc` for recognition models, `hmean-iou` for detection models.
- `--device`: Device for evaluation, `cuda:0` as default.

#### Results and Models


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

**Notes**:
- TensorRT upsampling operation is a little different from pytorch. For DBNet and PANet, we suggest replacing upsampling operations with neast mode to operations with bilinear mode. [Here](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpem_ffm.py#L33) for PANet, [here](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpn_cat.py#L111) and [here](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpn_cat.py#L121) for DBNet. As is shown in the above table, networks with tag * means the upsampling mode is changed.
- Note that, changing upsampling mode reduces less performance compared with using nearst mode. However, the weights of networks are trained through nearst mode. To persue best performance, using bilinear mode for both training and TensorRT deployment is recommanded.
- All ONNX and TensorRT models are evaluated with dynamic shape on the datasets and images are preprocessed according to the original config file.
- This tool is still experimental, and we only support `detection` and `recognition` for now.
