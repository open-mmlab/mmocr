# 推理

MMOCR 为示例和应用，以 [ocr.py](/mmocr/ocr.py) 脚本形式，提供了方便使用的 API。

该 API 可以通过命令行执行，也可以在 python 脚本内调用。在该 API 里，MMOCR 里的所有模型能以独立模块的形式被调用或串联。

```{warning}
该脚本仍在重构过程中，在接下来的版本接口中有可能会发生变化。
```

## 案例一：文本检测

<div align="center">
    <img src="https://user-images.githubusercontent.com/24622904/187825864-8ead5acb-c3c5-443b-bd90-3f4b188fa315.jpg"  height="250"/>
</div>

**注：** 使用 TextSnake 检测模型对图像上的文本进行检测，并保存可视化的文件。

- 命令行执行：

```shell
python mmocr/ocr.py demo/demo_text_det.jpg --det TextSnake --img-out-dir demo/
```

- Python 调用：

```python
from mmocr.ocr import MMOCR

# 导入模型到内存
ocr = MMOCR(det='TextSnake')

# 推理
results = ocr.readtext('demo/demo_text_det.jpg', img_out_dir='demo/')
```

## 案例二：文本检测+识别

<div align="center">
    <img src="https://user-images.githubusercontent.com/24622904/187825445-d30cbfa6-5549-4358-97fe-245f08f4ed94.jpg" height="250"/>
</div>

**注：** 使用 DB_r18 检测模型和 CRNN 识别模型，对 demo/demo_text_det.jpg 图片执行 ocr（检测+识别）推理，在终端打印结果并展示可视化结果。

- 命令行执行：

```shell
python mmocr/ocr.py --det DB_r18 --recog CRNN demo/demo_text_ocr.jpg --print-result --show
```

```{note}

当用户从命令行执行脚本时，默认配置文件都会保存在 `configs/` 目录下。用户可以通过指定 `config_dir` 的值来自定义读取配置文件的文件夹。

```

- Python 调用：

```python
from mmocr.ocr import MMOCR

# 导入模型到内存
ocr = MMOCR(det='DB_r18', recog='CRNN')

# 推理
results = ocr.readtext('demo/demo_text_ocr.jpg', print_result=True, show=True)
```

## 案例三： 文本检测+识别+关键信息提取

<div align="center">
    <img src="https://user-images.githubusercontent.com/24622904/187825451-6b043df9-10f7-4656-a528-45fe043df92b.jpg" height="250"/>
</div>

**注：** 首先，使用 DB_r18 检测模型和 CRNN 识别模型，进行端到端的 ocr （检测+识别）推理，然后对得到的结果，使用 SDMGR 模型提取关键信息（KIE），并展示可视化结果。

- 命令行执行：

```shell
python mmocr/ocr.py demo/demo_kie.jpeg  --det DB_r18 --recog CRNN --kie SDMGR --print-result --show
```

```{note}

当用户从命令行执行脚本时，默认配置文件都会保存在 `configs/` 目录下。用户可以通过指定 `config_dir` 的值来自定义读取配置文件的文件夹。

```

- Python 调用：

```python
from mmocr.ocr import MMOCR

# 导入模型到内存
ocr = MMOCR(det='DB_r18', recog='CRNN', kie='SDMGR')

# 推理
results = ocr.readtext('demo/demo_kie.jpeg', print_result=True, show=True)
```

## API 参数

该 API 有多个可供使用的参数列表。下表是 python 接口的参数。

**MMOCR():**

| 参数           | 类型               | 默认值   | 描述                                                                                     |
| -------------- | ------------------ | -------- | ---------------------------------------------------------------------------------------- |
| `det`          | 参考 **模型** 章节 | None     | 文本检测算法                                                                             |
| `recog`        | 参考 **模型** 章节 | None     | 文本识别算法                                                                             |
| `kie` \[1\]    | 参考 **模型** 章节 | None     | 关键信息提取算法                                                                         |
| `config_dir`   | str                | configs/ | 用于存放所有配置文件的文件夹路径                                                         |
| `det_config`   | str                | None     | 指定检测模型的自定义配置文件路径                                                         |
| `det_ckpt`     | str                | None     | 指定检测模型的自定义参数文件路径                                                         |
| `recog_config` | str                | None     | 指定识别模型的自定义配置文件路径                                                         |
| `recog_ckpt`   | str                | None     | 指定识别模型的自定义参数文件路径                                                         |
| `kie_config`   | str                | None     | 指定关键信息提取模型的自定义配置路径                                                     |
| `kie_ckpt`     | str                | None     | 指定关键信息提取的自定义参数文件路径                                                     |
| `device`       | str                | None     | 推理时使用的设备标识, 支持 `torch.device` 所包含的所有设备字符. 例如, 'cuda:0' 或 'cpu'. |

\[1\]: `kie` 当且仅当同时指定了文本检测和识别模型时才有效。

```{note}

mmocr 为了方便使用提供了预置的模型配置和对应的预训练权重，用户可以通过指定 `det` 和/或 `recog` 值来指定使用，这种方法等同于分别单独指定其对应的 `*_config` 和 `*_ckpt`。需要注意的是，手动指定 `*_config` 和 `*_ckpt` 会覆盖 `det` 和/或 `recog` 指定模型预置的配置和权重值。 同理 `kie`， `kie_config` 和 `kie_ckpt` 的参数设定逻辑相同。

```

### readtext()

| 参数           | 类型                    | 默认值   | 描述                                                                  |
| -------------- | ----------------------- | -------- | --------------------------------------------------------------------- |
| `img`          | str/list/tuple/np.array | **必填** | 图像，文件夹路径，np array 或 list/tuple （包含图片路径或 np arrays） |
| `img_out_dir`  | str                     | None     | 存放导出图片结果的文件夹                                              |
| `show`         | bool                    | False    | 是否在屏幕展示可视化结果                                              |
| `print_result` | bool                    | False    | 是否展示每个图片的结果                                                |

以上所有参数在命令行同样适用，只需要在参数前简单添加两个连接符，并且将下参数中的下划线替换为连接符即可。
（*例如：* `img_out_dir` 变成了 `--img-out-dir`）

对于布尔类型参数，添加在命令中默认为 true。
（*例如：* `python mmocr/demo/ocr.py --det DB_r18 demo/demo_text_det.jpg --print_result` 意为 `print_result` 的参数值设置为 `True`）

## 模型

**文本检测：**

| 名称          |                                                                           引用                                                                            |
| ------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DB_r18        |            [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#real-time-scene-text-detection-with-differentiable-binarization)            |
| DB_r50        |            [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#real-time-scene-text-detection-with-differentiable-binarization)            |
| DBPP_r50      |                                        [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#dbnetpp)                                        |
| DRRG          |                                         [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#drrg)                                          |
| FCE_IC15      |             [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)             |
| FCE_CTW_DCNv2 |             [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)             |
| MaskRCNN_CTW  |                                      [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#mask-r-cnn)                                       |
| MaskRCNN_IC15 |                                      [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#mask-r-cnn)                                       |
| PANet_CTW     | [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#efficient-and-accurate-arbitrary-shaped-text-detection-with-pixel-aggregation-network) |
| PANet_IC15    | [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#efficient-and-accurate-arbitrary-shaped-text-detection-with-pixel-aggregation-network) |
| PS_CTW        |                                        [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#psenet)                                         |
| PS_IC15       |                                        [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#psenet)                                         |
| TextSnake     |                                       [链接](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#textsnake)                                       |

**文本识别：**

| 名称 |                                                                                              引用                                                                                               |
| ---- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| CRNN | [链接](https://mmocr.readthedocs.io/en/dev-1.x/textrecog_models.html#an-end-to-end-trainable-neural-network-for-image-based-sequence-recognition-and-its-application-to-scene-text-recognition) |

**关键信息提取：**

| 名称                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------- |
| [SDMGR](https://mmocr.readthedocs.io/en/dev-1.x/kie_models.html#spatial-dual-modality-graph-reasoning-for-key-information-extraction) |

## 其他需要注意

- 执行检测+识别的推理（端到端 ocr），需要同时定义 `det` 和 `recog` 参数
- 如果只需要执行检测，则 `recog` 参数设置为 `None`。
- 如果只需要执行识别，则 `det` 参数设置为 `None`。

如果你对新特性有任何建议，请随时开一个 issue，甚至可以提一个 PR:)
