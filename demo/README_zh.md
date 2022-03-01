# 演示

MMOCR 为示例和应用，以 [ocr.py](https://github.com/open-mmlab/mmocr/blob/main/mmocr/utils/ocr.py) 脚本形式，提供了方便使用的 API。

该 API 可以通过命令行执行，也可以在 python 脚本内调用。

---

## 案例一：文本检测

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/text_det_pred.jpg"/><br>
</div>
<br>

**注：** 使用 TextSnake 检测模型对图像上的文本进行检测，结果用 json 格式的文件（默认）导出，并保存可视化的文件。

- 命令行执行：

```shell
python mmocr/utils/ocr.py demo/demo_text_det.jpg --output demo/det_out.jpg --det TextSnake --recog None --export demo/
```

- Python 调用：

```python
from mmocr.utils.ocr import MMOCR

# 导入模型到内存
ocr = MMOCR(det='TextSnake', recog=None)

# 推理
results = ocr.readtext('demo/demo_text_det.jpg', output='demo/det_out.jpg', export='demo/')
```

## 案例二：文本识别

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/text_recog_pred.jpg"/><br>
</div>
<br>

**注：** 使用 CRNN_TPS 识别模型对多张图片进行批量识别。*批处理的尺寸设置为 10，以防内存溢出引起的 CUDA 运行时错误。*


- 命令行执行：

```shell
python mmocr/utils/ocr.py %INPUT_FOLDER_PATH% --det None --recog CRNN_TPS --batch-mode --single-batch-size 10 --output %OUPUT_FOLDER_PATH%
```

- Python 调用：

```python
from mmocr.utils.ocr import MMOCR

# 导入模型到内存
ocr = MMOCR(det=None, recog='CRNN_TPS')

# 推理
results = ocr.readtext(%INPUT_FOLDER_PATH%, output = %OUTPUT_FOLDER_PATH%, batch_mode=True, single_batch_size = 10)
```

## 案例三：文本检测+识别

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/demo_ocr_pred.jpg"/><br>
</div>
<br>

**注：** 使用 PANet_IC15（默认）检测模型和 SAR（默认）识别模型，对 demo/demo_text_det.jpg 图片执行 ocr（检测+识别）推理，在终端打印结果并展示可视化结果。

- 命令行执行：

```shell
python mmocr/utils/ocr.py demo/demo_text_ocr.jpg --print-result --imshow
```

:::{note}

当从命令行调用脚本时，配置文件默认保存在 `configs`目录下，用户也可以设定通过设定 `config_dir` 来自定义配置文件读取

:::

- Python 调用：

```python
from mmocr.utils.ocr import MMOCR

# 导入模型到内存
ocr = MMOCR()

# 推理
results = ocr.readtext('demo/demo_text_ocr.jpg', print_result=True, imshow=True)
```

---

## 案例 4： 文本检测+识别+关键信息提取

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/demo_kie_pred.png"/><br>
</div>
<br>

**注：** 首先，使用 PS_CTW 检测模型和 SAR 识别模型，进行端到端的 ocr （检测+识别）推理，然后对得到的结果，使用 SDMGR 模型提取关键信息（KIE），并展示可视化结果。

- 命令行执行：

```shell
python mmocr/utils/ocr.py demo/demo_kie.jpeg  --det PS_CTW --recog SAR --kie SDMGR --print-result --imshow
```

:::{note}

当用户从命令行执行脚本时，会默认配置文件都保存在 `configs/` 目录下。用户可以通过指定 `config_dir` 的值来自定义读取配置文件的文件夹。

:::

- Python 调用：

```python
from mmocr.utils.ocr import MMOCR

# 导入模型到内存
ocr = MMOCR(det='PS_CTW', recog='SAR', kie='SDMGR')

# 推理
results = ocr.readtext('demo/demo_kie.jpeg', print_result=True, imshow=True)
```

---

## API 参数

该 API 有多个可供使用的参数列表。下表是 python 接口的参数。

**MMOCR():**

| 参数      | 类型                  | 默认值       | 描述                                                 |
| -------------- | --------------------- | ------------- | ----------------------------------------------------------- |
| `det`          | 参考 [models](#models) | PANet_IC15    | 文本检测算法                                    |
| `recog`        | 参考 [models](#models) | SAR           | 文本识别算法                                  |
| `kie` [1]      | 参考 [models](#models) | None          | 关键信息提取算法                                  |
| `config_dir`   | str                   | configs/      | 用于存放所有配置文件的文件夹路径  |
| `det_config`   | str                   | None          | 指定检测模型的自定义配置文件路径         |
| `det_ckpt`     | str                   | None          | 指定检测模型的自定义参数文件路径         |
| `recog_config` | str                   | None          | 指定识别模型的自定义配置文件路径 |
| `recog_ckpt`   | str                   | None          | 指定识别模型的自定义参数文件路径 |
| `kie_config`   | str                   | None          | 指定关键信息提取模型的自定义配置路径 |
| `kie_ckpt`     | str                   | None          | 指定关键信息提取的自定义参数文件路径 |
| `device`       | str                   | None        | 推理时使用的设备标识, 支持 `torch.device` 所包含的所有设备字符. 例如, 'cuda:0' 或 'cpu'. |

[1]: `kie` 当且仅当同时指定了文本检测和识别模型时才有效。

:::{note}

mmocr 为了方便使用提供了预置的模型配置和对应的预训练权重，用户可以通过指定 `det` 和/或 `recog` 值来指定使用，这种方法等同于分别单独指定其对应的 `*_config` 和 `*_ckpt`。需要注意的是，手动指定 `*_config` 和 `*_ckpt` 会覆盖 `det` 和/或 `recog` 指定模型预置的配置和权重值。 同理 `kie`， `kie_config` 和 `kie_ckpt` 的参数设定逻辑相同。

:::

### readtext()

| 参数           | 类型                    | 默认值      | 描述                                                            |
| ------------------- | ----------------------- | ------------ | ---------------------------------------------------------------------- |
| `img`               | str/list/tuple/np.array | **必填** | 图像，文件夹路径，np array 或 list/tuple （包含图片路径或 np arrays） |
| `output`           | str                     | None         | 可视化输出结果 - 图片路径或文件夹路径                  |
| `batch_mode`        | bool                    | False        | 是否使用批处理模式推理 [1]                                  |
| `det_batch_size`    | int                     | 0            | 文本检测的批处理大小（设置为 0 则与待推理图片个数相同）                         |
| `recog_batch_size`  | int                     | 0            | 文本识别的批处理大小（设置为 0 则与待推理图片个数相同）                       |
| `single_batch_size` | int                     | 0            | 仅用于检测或识别使用的批处理大小                           |
| `export`            | str                     | None         | 存放导出图片结果的文件夹                    |
| `export_format`     | str                     | json         | 导出的结果文件格式                                  |
| `details`           | bool                    | False        | 是否包含文本框的坐标和置信度的值       |
| `imshow`            | bool                    | False        | 是否在屏幕展示可视化结果                     |
| `print_result`      | bool                    | False        | 是否展示每个图片的结果                              |
| `merge`             | bool                    | False        | 是否对相邻框进行合并 [2]                              |
| `merge_xdist`       | float                   | 20           | 合并相邻框的最大x-轴距离                              |

[1]: `batch_mode` 需确保模型兼容批处理模式（见下表模型是否支持批处理）。

[2]: `merge` 只有同时运行检测+识别模式，参数才有效。

以上所有参数在命令行同样适用，只需要在参数前简单添加两个连接符，并且将下参数中的下划线替换为连接符即可。
（*例如：* `det_batch_size` 变成了 `--det-batch-size`）

对于布尔类型参数，添加在命令中默认为true。
（*例如：* `python mmocr/utils/ocr.py demo/demo_text_det.jpg --batch_mode --print_result` 意为 `batch_mode` 和 `print_result` 的参数值设置为 `True`）

---

## 模型

**文本检测：**

| 名称          | `batch_mode` 推理支持 |
| ------------- | :------------------: |
| [DB_r18](https://mmocr.readthedocs.io/en/latest/textdet_models.html#real-time-scene-text-detection-with-differentiable-binarization)            |         :x:          |
| [DB_r50](https://mmocr.readthedocs.io/en/latest/textdet_models.html#real-time-scene-text-detection-with-differentiable-binarization)            |         :x:          |
| [DRRG](https://mmocr.readthedocs.io/en/latest/textdet_models.html#drrg)                                          |         :x:          |
| [FCE_IC15](https://mmocr.readthedocs.io/en/latest/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)             |         :x:          |
| [FCE_CTW_DCNv2](https://mmocr.readthedocs.io/en/latest/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)             |         :x:          |
| [MaskRCNN_CTW](https://mmocr.readthedocs.io/en/latest/textdet_models.html#mask-r-cnn)                                       |         :x:          |
| [MaskRCNN_IC15](https://mmocr.readthedocs.io/en/latest/textdet_models.html#mask-r-cnn)                                       |         :x:          |
| [MaskRCNN_IC17](https://mmocr.readthedocs.io/en/latest/textdet_models.html#mask-r-cnn)                                       |         :x:          |
| [PANet_CTW](https://mmocr.readthedocs.io/en/latest/textdet_models.html#efficient-and-accurate-arbitrary-shaped-text-detection-with-pixel-aggregation-network) |  :heavy_check_mark:  |
| [PANet_IC15](https://mmocr.readthedocs.io/en/latest/textdet_models.html#efficient-and-accurate-arbitrary-shaped-text-detection-with-pixel-aggregation-network) |  :heavy_check_mark:  |
| [PS_CTW](https://mmocr.readthedocs.io/en/latest/textdet_models.html#psenet)                                         |         :x:          |
| [PS_IC15](https://mmocr.readthedocs.io/en/latest/textdet_models.html#psenet)                                         |         :x:          |
| [TextSnake](https://mmocr.readthedocs.io/en/latest/textdet_models.html#textsnake)                                       |  :heavy_check_mark:  |

**文本识别：**

| 名称          | `batch_mode` 推理支持 |
| ------------- |:------------------: |
| [ABINet](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#read-like-humans-autonomous-bidirectional-and-iterative-language-modeling-for-scene-text-recognition) |         :heavy_check_mark:          |
| [CRNN](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#an-end-to-end-trainable-neural-network-for-image-based-sequence-recognition-and-its-application-to-scene-text-recognition) |         :x:          |
| [SAR](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition) |  :heavy_check_mark:  |
| [SAR_CN](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition) |  :heavy_check_mark:  |
| [NRTR_1/16-1/8](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#nrtr) |  :heavy_check_mark:  |
| [NRTR_1/8-1/4](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#nrtr) |  :heavy_check_mark:  |
| [RobustScanner](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#robustscanner-dynamically-enhancing-positional-clues-for-robust-text-recognition) |  :heavy_check_mark:  |
| [SATRN](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#satrn) |  :heavy_check_mark:  |
| [SATRN_sm](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#satrn) |  :heavy_check_mark:  |
| [SEG](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#segocr-simple-baseline) |         :x:          |
| [CRNN_TPS](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#crnn-with-tps-based-stn) |  :heavy_check_mark:  |

:::{note}

SAR_CN 是唯一支持中文字符识别的模型，并且它需要一个中文字典。以便推理能成功运行，请先从 [这里](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#chinese-dataset) 下载辞典。

:::

**关键信息提取：**

| 名称          | `batch_mode` 支持 |
| ------------- | :------------------: |
| [SDMGR](https://mmocr.readthedocs.io/en/latest/kie_models.html#spatial-dual-modality-graph-reasoning-for-key-information-extraction)            |         :heavy_check_mark:          |
---

## 其他需要注意

- 执行检测+识别的推理（端到端 ocr），需要同时定义 `det` 和 `recog` 参数
- 如果只需要执行检测，则 `recog` 参数设置为 `None`。
- 如果只需要执行识别，则 `det` 参数设置为 `None`。
- `details` 参数仅在端到端的 ocr 模型有效。
- `det_batch_size` 和 `recog_batch_size` 指定了在同时间传递给模型的图片数量。为了提高推理速度，应该尽可能设置你能设置的最大值。最大的批处理值受模型复杂度和 GPU 的显存大小限制。

如果你对新特性有任何建议，请随时开一个 issue，甚至可以提一个 PR:)
