## 部署

我们在 `tools/deployment` 目录下提供了一些部署工具。

### 转换至 ONNX (试验性的)

我们提供了将模型转换至 [ONNX](https://github.com/onnx/onnx) 格式的脚本。转换后的模型可以使用诸如 [Netron](https://github.com/lutzroeder/netron) 的工具可视化。 此外，我们也支持比较 PyTorch 和 ONNX 模型的输出结果。

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

所有参数的说明：

| 参数               | 类型           | 描述                                                         |
| ------------------ | -------------- | ------------------------------------------------------------ |
| `model_config`     | str            | 模型配置文件的路径。                                         |
| `model_ckpt`       | str            | 模型权重文件的路径。                                         |
| `model_type`       | 'recog', 'det' | 配置文件对应的模型类型。                                     |
| `image_path`       | str            | 输入图片的路径。                                             |
| `--output-file`    | str            | 输出的 ONNX 模型路径。 默认为 `tmp.onnx`。                   |
| `--device-id`      | int            | 使用哪块 GPU。默认为0。                                      |
| `--opset-version`  | int            | ONNX 操作集版本。默认为11。                                  |
| `--verify`         | bool           | 决定是否验证输出模型的正确性。默认为 `False`。               |
| `--verbose`        | bool           | 决定是否打印导出模型的结构，默认为 `False`。                 |
| `--show`           | bool           | 决定是否可视化 ONNXRuntime 和 PyTorch 的输出。默认为 `False`。 |
| `--dynamic-export` | bool           | 决定是否导出有动态输入和输出尺寸的 ONNX 模型。默认为 `False`。 |

:::{note}
 这个工具仍然是试验性的。一些定制的操作没有被支持，并且我们目前仅支持一部分的文本检测和文本识别算法。
:::

#### 支持导出到 ONNX 的模型列表

下表列出的模型可以保证导出到 ONNX 并且可以在 ONNX Runtime 下运行。

|  模型 |                                                                      配置                                                                      | 动态尺寸 | 批推理 | 注 |
|:------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|:---------------:|:----:|
|  DBNet |    [dbnet_r18_fpnc_1200e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py)    |       Y       |        N        |      |
| PSENet |      [psenet_r50_fpnf_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py)     |       Y       |        Y        |      |
| PSENet |    [psenet_r50_fpnf_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py)   |       Y       |        Y        |      |
|  PANet |   [panet_r18_fpem_ffm_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py)   |       Y       |        Y        |      |
|  PANet | [panet_r18_fpem_ffm_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |       Y       |        Y        |      |
|  CRNN  |             [crnn_academic_dataset.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/crnn_academic_dataset.py)            |       Y       |        Y        | CRNN 仅接受高度为32的输入 |

:::{note}
- *以上所有模型测试基于PyTorch==1.8.1，onnxruntime==1.7.0进行*
- 如果你在上述模型中遇到问题，请创建一个issue，我们会尽快处理。
- 因为这个特性是试验性的，可能变动很快，请尽量使用最新版的 `mmcv` 和 `mmocr` 尝试。
:::

###  ONNX 转 TensorRT （试验性的）

我们也提供了从 [ONNX](https://github.com/onnx/onnx) 模型转换至 [TensorRT](https://github.com/NVIDIA/TensorRT) 格式的脚本。另外，我们支持比较 ONNX 和 TensorRT 模型的输出结果。


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

所有参数的说明：

| 参数               | 类型           | 描述                                                         |
| ------------------ | -------------- | ------------------------------------------------------------ |
| `model_config`     | str            | 模型配置文件的路径。                                         |
| `model_type`       | 'recog', 'det' | 配置文件对应的模型类型。                                     |
| `image_path`       | str            | 输入图片的路径。                                             |
| `onnx_file`        | str            | 输入的 ONNX 文件路径。                                       |
| `--trt-file`       | str            | 输出的 TensorRT 模型路径。默认为 `tmp.trt`。                 |
| `--max-shape`      | int * 4        | 模型输入的最大尺寸。                                         |
| `--min-shape`      | int * 4        | 模型输入的最小尺寸。                                         |
| `--workspace-size` | int            | 最大工作空间大小，单位为 GiB。默认为1。                      |
| `--fp16`           | bool           | 决定是否输出 fp16 模式的 TensorRT 模型。默认为 `False`。     |
| `--verify`         | bool           | 决定是否验证输出模型的正确性。默认为 `False`。               |
| `--show`           | bool           | 决定是否可视化 ONNX 和 TensorRT 的输出。默认为 `False`。     |
| `--verbose`        | bool           | 决定是否在创建 TensorRT 引擎时打印日志信息。默认为 `False`。 |

:::{note}
 这个工具仍然是试验性的。一些定制的操作模型没有被支持。我们目前仅支持一部的文本检测和文本识别算法。
:::

#### 支持导出到 TensorRT 的模型列表

下表列出的模型可以保证导出到 TensorRT 引擎并且可以在 TensorRT 下运行。

|  模型 |                                                                      配置                                                                      | 动态尺寸 | 批推理 | 注 |
|:------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|:---------------:|:----:|
|  DBNet |    [dbnet_r18_fpnc_1200e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py)    |       Y       |        N        |      |
| PSENet |      [psenet_r50_fpnf_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py)     |       Y       |        Y        |      |
| PSENet |    [psenet_r50_fpnf_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py)   |       Y       |        Y        |      |
|  PANet |   [panet_r18_fpem_ffm_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py)   |       Y       |        Y        |      |
|  PANet | [panet_r18_fpem_ffm_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |       Y       |        Y        |      |
|  CRNN  |             [crnn_academic_dataset.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/crnn_academic_dataset.py)            |       Y       |        Y        | CRNN 仅接受高度为32的输入 |

:::{note}
- *以上所有模型测试基于PyTorch==1.8.1，onnxruntime==1.7.0，tensorrt==7.2.1.6进行*
- 如果你在上述模型中遇到问题，请创建一个 issue，我们会尽快处理。
- 因为这个特性是试验性的，可能变动很快，请尽量使用最新版的 `mmcv` 和 `mmocr` 尝试。
:::


### 评估 ONNX 和 TensorRT 模型（试验性的）

我们在 `tools/deployment/deploy_test.py ` 中提供了评估 TensorRT 和 ONNX 模型的方法。

#### 前提条件
在评估 ONNX 和 TensorRT 模型之前，首先需要安装 ONNX，ONNXRuntime 和 TensorRT。根据 [ONNXRuntime in mmcv](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) 和 [TensorRT plugin in mmcv](https://github.com/open-mmlab/mmcv/blob/master/docs/tensorrt_plugin.md) 安装 ONNXRuntime 定制操作和 TensorRT 插件。

#### 使用

```bash
python tools/deploy_test.py \
    ${CONFIG_FILE} \
    ${MODEL_PATH} \
    ${MODEL_TYPE} \
    ${BACKEND} \
    --eval ${METRICS} \
    --device ${DEVICE}
```

#### 所有参数的说明

| 参数           | 类型                      | 描述                                                   |
| -------------- | ------------------------- | ------------------------------------------------------ |
| `model_config` | str                       | 模型配置文件的路径。                                   |
| `model_file`   | str                       | TensorRT 或 ONNX 模型路径。                            |
| `model_type`   | 'recog', 'det'            | 部署检测还是识别模型。                                 |
| `backend`      | 'TensorRT', 'ONNXRuntime' | 测试后端。                                             |
| `--eval`       | 'acc', 'hmean-iou'        | 评估指标。“acc”用于识别模型，“hmean-iou”用于检测模型。 |
| `--device`     | str                       | 评估使用的设备。默认为 `cuda:0`。                      |

#### 结果和模型

<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8">模型</th>
    <th class="tg-9wq8">配置</th>
    <th class="tg-9wq8">数据集</th>
    <th class="tg-9wq8">指标</th>
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
- TensorRT 上采样（upsample）操作和 PyTorch 有一点不同。对于 DBNet 和 PANet，我们建议把上采样的最近邻 (nearest) 模式代替成双线性 (bilinear) 模式。 PANet 的替换处在[这里](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpem_ffm.py#L33) ，DBNet 的替换处在[这里](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpn_cat.py#L111) 和 [这里](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpn_cat.py#L121) 。如在上表中显示的，带有标记*的网络的上采样模式均被改变了。
- 注意到，相比最近邻模式，使用更改后的上采样模式会降低性能。然而，默认网络的权重是通过最近邻（nearest）模式训练的。为了保持在部署中的最佳性能，建议在训练和 TensorRT 部署中使用双线性（bilinear）模式。
- 所有 ONNX 和 TensorRT 模型都使用数据集上的动态尺寸进行评估，图像根据原始配置文件进行预处理。
- 这个工具仍然是试验性的。一些定制的操作模型没有被支持。并且我们目前仅支持一部分的文本检测和文本识别算法。
:::
