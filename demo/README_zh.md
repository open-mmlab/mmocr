# 演示

我们提供了易于使用的 API，方便演示和应用，见 [ocr.py](https://github.com/open-mmlab/mmocr/blob/main/mmocr/utils/ocr.py) 脚本。


该 API 可以通过命令行调用，也可以通过另一个 python 脚本调用。

---

## 案例一：文本检测

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/text_det_pred.jpg"/><br>
</div>
<br>

**Instruction:** Perform detection inference on an image with the TextSnake recognition model, export the result in a json file (default) and save the visualization file.

**说明：** 使用 TextSnake 识别模型对图像上的文本进行检测推理，通过 json 格式的文件（默认）导出结果，并保存可视化文件。


- 命令行接口：

```shell
python mmocr/utils/ocr.py demo/demo_text_det.jpg --output demo/det_out.jpg --det TextSnake --recog None --export demo/
```

- Python 接口：

```python
from mmocr.utils.ocr import MMOCR

# Load models into memory
ocr = MMOCR(det='TextSnake', recog=None)

# Inference
results = ocr.readtext('demo/demo_text_det.jpg', output='demo/det_out.jpg', export='demo/')
```

## 案例二：文本识别

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/text_recog_pred.jpg"/><br>
</div>
<br>

**Instruction:** Perform batched recognition inference on a folder with hundreds of image with the CRNN_TPS recognition model and save the visualization results in another folder.
*Batch size is set to 10 to prevent out of memory CUDA runtime errors.*

**说明：** 使用 CRNN_TPS 识别模型对文件夹下成百上千张图片进行批量识别推理。*批处理的尺寸设置为 10，以防内存溢出引起的 CUDA 运行时错误。*


- 命令行接口：

```shell
python mmocr/utils/ocr.py %INPUT_FOLDER_PATH% --det None --recog CRNN_TPS --batch-mode --single-batch-size 10 --output %OUPUT_FOLDER_PATH%
```

- Python 接口：

```python
from mmocr.utils.ocr import MMOCR

# 导入模型到内存
ocr = MMOCR(det=None, recog='CRNN_TPS')

# 接口
results = ocr.readtext(%INPUT_FOLDER_PATH%, output = %OUTPUT_FOLDER_PATH%, batch_mode=True, single_batch_size = 10)
```

## 案例三：文本检测+识别

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/demo_ocr_pred.jpg"/><br>
</div>
<br>

**说明：** 使执行 ocr（检测+识别）接口，作用在 demo/demo_text_det.jpg 图片上，使用 PANet_IC15（默认）检测模型和 SAR（默认）识别模型，打印结果到终端并显示可视化结果。

- 命令行接口：

```shell
python mmocr/utils/ocr.py demo/demo_text_ocr.jpg --print-result --imshow
```

:::{注意}

当从命令行调用脚本时，脚本假设配置文件保存在 `configs/` 目录下。用户可可以通过指定 `config_dir` 的值自定义文件夹。

:::

- Python 接口:

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

**Instruction:** Perform end-to-end ocr (det + recog) inference first with PS_CTW detection model and SAR recognition model, then run KIE inference with SDMGR model on the ocr result and show the visualization.
**说明：** 首先，使用 PS_CTW 检测模型和 SAR 识别模型，执行端到端 ocr （检测+识别）推理，然后对之前的结果使用 SDMGR 模型提取关键信息（KIE），并展示可视化结果。


- 命令行接口：

```shell
python mmocr/utils/ocr.py demo/demo_kie.jpeg  --det PS_CTW --recog SAR --kie SDMGR --print-result --imshow
```

:::{注意}

当从命令行调用脚本时，脚本假设配置文件保存在 `configs/` 目录下。用户可可以通过指定 `config_dir` 的值自定义文件夹。

:::

- Python 接口：

```python
from mmocr.utils.ocr import MMOCR

# 导入模型到内存
ocr = MMOCR(det='PS_CTW', recog='SAR', kie='SDMGR')

# 推理
results = ocr.readtext('demo/demo_kie.jpeg', print_result=True, imshow=True)
```

---

## API 参数


该 API 有多种可使用的参数。下表时 python 接口的参数。


**MMOCR():**

| 参数      | 类型                  | 默认值       | 描述                                                 |
| -------------- | --------------------- | ------------- | ----------------------------------------------------------- |
| `det`          | 参考 [models](#models) | PANet_IC15    | 文本检测算法                                    |
| `recog`        | 参考 [models](#models) | SAR           | 文本识别算法                                  |
| `kie` [1]      | 参考 [models](#models) | None          | 关键信息提取算法                                  |
| `config_dir`   | str                   | configs/      | 包含模型所有配置文件的文件的路径  |
| `det_config`   | str                   | None          | 指定检测模型的自定义配置文件路径         |
| `det_ckpt`     | str                   | None          | 指定检测模型的自定义参数文件路径         |
| `recog_config` | str                   | None          | 指定识别模型的自定义配置文件路径 |
| `recog_ckpt`   | str                   | None          | 指定识别模型的自定义参数文件路径 |
| `kie_config`   | str                   | None          | 指定关键信息提取模型的自定义配置路径 |
| `kie_ckpt`     | str                   | None          | 指定关键信息提取的自定义参数文件路径 |
| `device`       | str                   | None        | 推理时用的设备标识, 支持 `torch.device` 所包含的所有设备字符. 例如, 'cuda:0' 或 'cpu'. |

[1]: `kie` 当且仅当同时指定了文本检测和识别模型时才有效。

:::{注意}

User can use default pretrained models by specifying `det` and/or `recog`, which is equivalent to specifying their corresponding `*_config` and `*_ckpt`. However, manually specifying `*_config` and `*_ckpt` will always override values set by `det` and/or `recog`. Similar rules also apply to `kie`, `kie_config` and `kie_ckpt`.

:::

### readtext()

| Arguments           | Type                    | Default      | Description                                                            |
| ------------------- | ----------------------- | ------------ | ---------------------------------------------------------------------- |
| `img`               | str/list/tuple/np.array | **required** | img, folder path, np array or list/tuple (with img paths or np arrays) |
| `output`           | str                     | None         | Output result visualization - img path or folder path                  |
| `batch_mode`        | bool                    | False        | Whether use batch mode for inference [1]                                  |
| `det_batch_size`    | int                     | 0            | Batch size for text detection (0 for max size)                         |
| `recog_batch_size`  | int                     | 0            | Batch size for text recognition (0 for max size)                       |
| `single_batch_size` | int                     | 0            | Batch size for only detection or recognition                           |
| `export`            | str                     | None         | Folder where the results of each image are exported                    |
| `export_format`     | str                     | json         | Format of the exported result file(s)                                  |
| `details`           | bool                    | False        | Whether include the text boxes coordinates and confidence values       |
| `imshow`            | bool                    | False        | Whether to show the result visualization on screen                     |
| `print_result`      | bool                    | False        | Whether to show the result for each image                              |
| `merge`             | bool                    | False        | Whether to merge neighboring boxes [2]                              |
| `merge_xdist`       | float                   | 20           | The maximum x-axis distance to merge boxes                              |

[1]: Make sure that the model is compatible with batch mode.

[2]: Only effective when the script is running in det + recog mode.

All arguments are the same for the cli, all you need to do is add 2 hyphens at the beginning of the argument and replace underscores by hyphens.
(*Example:* `det_batch_size` becomes `--det-batch-size`)

For bool type arguments, putting the argument in the command stores it as true.
(*Example:* `python mmocr/utils/ocr.py demo/demo_text_det.jpg --batch_mode --print_result`
means that `batch_mode` and `print_result` are set to `True`)

---

## Models

**Text detection:**

| Name          |                                                                        Reference                                                                         | `batch_mode` inference support |
| ------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------: |
| DB_r18        |            [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#real-time-scene-text-detection-with-differentiable-binarization)            |         :x:          |
| DB_r50        |            [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#real-time-scene-text-detection-with-differentiable-binarization)            |         :x:          |
| DRRG          |                                         [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#drrg)                                          |         :x:          |
| FCE_IC15      |             [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)             |         :x:          |
| FCE_CTW_DCNv2 |             [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)             |         :x:          |
| MaskRCNN_CTW  |                                      [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#mask-r-cnn)                                       |         :x:          |
| MaskRCNN_IC15 |                                      [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#mask-r-cnn)                                       |         :x:          |
| MaskRCNN_IC17 |                                      [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#mask-r-cnn)                                       |         :x:          |
| PANet_CTW     | [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#efficient-and-accurate-arbitrary-shaped-text-detection-with-pixel-aggregation-network) |  :heavy_check_mark:  |
| PANet_IC15    | [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#efficient-and-accurate-arbitrary-shaped-text-detection-with-pixel-aggregation-network) |  :heavy_check_mark:  |
| PS_CTW        |                                        [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#psenet)                                         |         :x:          |
| PS_IC15       |                                        [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#psenet)                                         |         :x:          |
| TextSnake     |                                       [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#textsnake)                                       |  :heavy_check_mark:  |

**Text recognition:**

| Name          |                                                             Reference                                                              | `batch_mode` inference support |
| ------------- | :--------------------------------------------------------------------------------------------------------------------------------: | :------------------: |
| ABINet          | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#read-like-humans-autonomous-bidirectional-and-iterative-language-modeling-for-scene-text-recognition) |         :heavy_check_mark:          |
| CRNN          | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#an-end-to-end-trainable-neural-network-for-image-based-sequence-recognition-and-its-application-to-scene-text-recognition) |         :x:          |
| SAR           | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition) |  :heavy_check_mark:  |
| SAR_CN *          | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition) |  :heavy_check_mark:  |
| NRTR_1/16-1/8 | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#nrtr) |  :heavy_check_mark:  |
| NRTR_1/8-1/4  | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#nrtr) |  :heavy_check_mark:  |
| RobustScanner | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#robustscanner-dynamically-enhancing-positional-clues-for-robust-text-recognition) |  :heavy_check_mark:  |
| SATRN | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#satrn) |  :heavy_check_mark:  |
| SATRN_sm | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#satrn) |  :heavy_check_mark:  |
| SEG           | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#segocr-simple-baseline) |         :x:          |
| CRNN_TPS      | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#crnn-with-tps-based-stn) |  :heavy_check_mark:  |

:::{warning}

SAR_CN is the only model that supports Chinese character recognition and it requires
a Chinese dictionary. Please download the dictionary from [here](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#chinese-dataset) for a successful run.

:::

**Key information extraction:**

| Name          |                                                                        Reference                                                                         | `batch_mode` support |
| ------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------: |
| SDMGR         |            [link](https://mmocr.readthedocs.io/en/latest/kie_models.html#spatial-dual-modality-graph-reasoning-for-key-information-extraction)            |         :heavy_check_mark:          |
---

## Additional info

- To perform det + recog inference (end2end ocr), both the `det` and `recog` arguments must be defined.
- To perform only detection set the `recog` argument to `None`.
- To perform only recognition set the `det` argument to `None`.
- `details` argument only works with end2end ocr.
- `det_batch_size` and `recog_batch_size` arguments define the number of images you want to forward to the model at the same time. For maximum speed, set this to the highest number you can. The max batch size is limited by the model complexity and the GPU VRAM size.

If you have any suggestions for new features, feel free to open a thread or even PR :)
