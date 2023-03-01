# 推理器

在 OpenMMLab 中，所有的推理操作都被统一到了推理器 `Inferencer` 中。推理器被设计成为一个简洁易用的 API，它在不同的 OpenMMLab 库中都有着非常相似的接口。

MMOCR 中存在两种不同的推理器：

- **标准推理器**：MMOCR 中的每个基本任务都有一个标准推理器，即 `TextDetInferencer`，`TextRecInferencer`，`TextSpottingInferencer` 和 `KIEInferencer`。它们具有非常相似的接口，具有标准的输入/输出协议，并且总体遵循 OpenMMLab 的设计。这些推理器也可以被串联在一起，以便对一系列任务进行推理。
- **MMOCRInferencer**：我们还提供了 `MMOCRInferencer`，一个专门为 MMOCR 设计的便捷推理接口。它封装和链接了 MMOCR 中的所有推理器，因此用户可以使用此推理器对图像执行一系列任务，并直接获得最终结果。*但是，它的接口与标准推理器有一些不同，并且为了简单起见，可能会牺牲一些标准的推理器功能。*

对于新用户，我们建议使用 [`MMOCRInferencer`](../user_guides/inference.md) 来测试不同模型的组合。

如果你是开发人员并希望将模型集成到自己的项目中，我们建议使用标准推理器，因为它们更灵活且标准化，并具有完整的功能。

## 基础用法

`````{tabs}

````{group-tab} MMOCRInferencer

为了便于使用，`MMOCRInferencer` 向用户提供了 Python 接口和命令行接口。例如，如果你想要对 demo/demo_text_ocr.jpg 进行 OCR，使用 `DBNet` 作为文本检测模型，`CRNN` 作为文本识别模型，只需执行以下命令:

::::{tabs}

:::{code-tab} python
>>> from mmocr.apis import MMOCRInferencer
>>> # 读取模型
>>> ocr = MMOCRInferencer(det='DBNet', rec='SAR')
>>> # 进行推理并可视化结果
>>> ocr('demo/demo_text_ocr.jpg', show=True)
:::

:::{code-tab} bash 命令行
python tools/infer.py demo/demo_text_ocr.jpg --det DBNet --rec SAR --show
:::
::::

可视化结果将被显示在一个新窗口中：

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/220563262-e9c1ab52-9b96-4d9c-bcb6-f55ff0b9e1be.png" height="250"/>
</div>

```{note}
如果你在没有 GUI 的服务器上运行 MMOCR，或者是通过禁用 X11 转发的 SSH 隧道运行该指令，`show`  选项将不起作用。然而，你仍然可以通过设置 `out_dir` 和 `save_vis=True` 参数将可视化数据保存到文件。阅读 [获取结果](#获取结果) 了解详情。
```

根据初始化参数，`MMOCRInferencer`可以在不同模式下运行。例如，如果初始化时指定了 `det`、`rec` 和 `kie`，它可以在 KIE 模式下运行。

::::{tabs}

:::{code-tab} python
>>> kie = MMOCRInferencer(det='DBNet', rec='SAR', kie='SDMGR')
>>> kie('demo/demo_kie.jpeg', show=True)
:::

:::{code-tab} bash 命令行
python tools/infer.py demo/demo_kie.jpeg --det DBNet --rec SAR --kie SDMGR --show
:::

::::

可视化结果如下：

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/220569700-fd4894bc-f65a-405e-95e7-ebd2d614aedd.png" height="250"/>
</div>
<br />

````

````{group-tab} 标准推理器

通常，OpenMMLab 中的所有标准推理器都具有非常相似的接口。下面的例子展示了如何使用 `TextDetInferencer` 对单个图像进行推理。

```python
>>> from mmocr.apis import TextDetInferencer
>>> # 读取模型
>>> inferencer = TextDetInferencer(model='DBNet')
>>> # 推理
>>> inferencer('demo/demo_text_ocr.jpg', show=True)
```

可视化结果如图：

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/221418215-2431d0e9-e16e-4deb-9c52-f8b86801706a.png" height="250"/>
</div>

可以见到，MMOCRInferencer 的 Python 接口与命令行接口的使用方法非常相似。下文将以 Python 接口为例，介绍 MMOCRInferencer 的具体用法。关于命令行接口的更多信息，请参考 [命令行接口](#命令行接口)。


````

`````

## 初始化

每个推理器必须使用一个模型进行初始化，也可以手动选择推理设备。

### 模型初始化

`````{tabs}

````{group-tab} MMOCRInferencer

对于每个任务，`MMOCRInferencer` 需要两个参数 `xxx` 和 `xxx_weights` （例如 `det` 和 `det_weights`）以对模型进行初始化。此处将以`det`和`det_weights`为例来说明一些典型的初始化模型的方法。

- 要用 MMOCR 的预训练模型进行推理，只需要把它的名字传给参数 `det`，权重将自动从 OpenMMLab 的模型库中下载和加载。[此处](../modelzoo.md#权重)记录了 MMOCR 中可以通过该方法初始化的所有模型。

  ```python
  >>> MMOCRInferencer(det='DBNet')
  ```

- 要加载自定义的配置和权重，你可以把配置文件的路径传给 `det`，把权重的路径传给 `det_weights`。

  ```python
  >>> MMOCRInferencer(det='path/to/dbnet_config.py', det_weights='path/to/dbnet.pth')
  ```

如果需要查看更多的初始化方法，请点击“标准推理器”选项卡。

````

````{group-tab} 标准推理器

每个标准的 `Inferencer` 都接受两个参数，`model` 和 `weights` 。在 MMOCRInferencer 中，这两个参数分别对应 `xxx` 和 `xxx_weights` （例如 `det` 和 `det_weights`）。

- `model` 接受模型的名称或配置文件的路径作为输入。模型的名称从 [model-index.yml](https://github.com/open-mmlab/mmocr/blob/1.x/model-index.yml) 中的模型的元文件（[示例](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/metafile.yml) ）中获取。你可以在[此处](../modelzoo.md#权重)找到可用权重的列表。

- `weights` 接受权重文件的路径。

<br />

此处列举了一些常见的初始化模型的方法。

- 你可以通过传递模型的名称给 `model` 来推理 MMOCR 的预训练模型。权重将会自动从 OpenMMLab 的模型库中下载并加载。

  ```python
  >>> from mmocr.apis import TextDetInferencer
  >>> inferencer = TextDetInferencer(model='DBNet')
  ```

  ```{note}
  模型与推理器的任务种类必须匹配。
  ```

  你可以通过将权重的路径或 URL 传递给 `weights` 来让推理器加载自定义的权重。

  ```python
  >>> inferencer = TextDetInferencer(model='DBNet', weights='path/to/dbnet.pth')
  ```

- 如果有自定义的配置和权重，你可以将配置文件的路径传递给 `model`，将权重的路径传递给 `weights`。

  ```python
  >>> inferencer = TextDetInferencer(model='path/to/dbnet_config.py', weights='path/to/dbnet.pth')
  ```

- 默认情况下，[MMEngine](https://github.com/open-mmlab/mmengine/) 会在训练模型时自动将配置文件转储到权重文件中。如果你有一个在 MMEngine 上训练的权重，你也可以将权重文件的路径传递给 `weights`，而不需要指定 `model`：

  ```python
  >>> # 如果无法在权重中找到配置文件，则会引发错误
  >>> inferencer = TextDetInferencer(weights='path/to/dbnet.pth')
  ```

- 传递配置文件到 `model` 而不指定 `weight` 则会产生一个随机初始化的模型。

````
`````

### 推理设备

每个推理器实例都会跟一个设备绑定。默认情况下，最佳设备是由 [MMEngine](https://github.com/open-mmlab/mmengine/) 自动决定的。你也可以通过指定 `device` 参数来改变设备。例如，你可以使用以下代码在 GPU 1上创建一个推理器。

`````{tabs}

````{group-tab} MMOCRInferencer

```python
>>> inferencer = MMOCRInferencer(det='DBNet', device='cuda:1')
```

````

````{group-tab} 标准推理器

```python
>>> inferencer = TextDetInferencer(model='DBNet', device='cuda:1')
```

````

`````

如要在 CPU 上创建一个推理器：

`````{tabs}

````{group-tab} MMOCRInferencer

```python
>>> inferencer = MMOCRInferencer(det='DBNet', device='cpu')
```

````

````{group-tab} 标准推理器

```python
>>> inferencer = TextDetInferencer(model='DBNet', device='cpu')
```

````

`````

请参考 [torch.device](torch.device) 了解 `device` 参数支持的所有形式。

## 推理

当推理器初始化后，你可以直接传入要推理的原始数据，从返回值中获取推理结果。

### 输入

`````{tabs}

````{tab} MMOCRInferencer / TextDetInferencer / TextRecInferencer / TextSpottingInferencer

输入可以是以下任意一种格式：

- str: 图像的路径/URL。

  ```python
  >>> inferencer('demo/demo_text_ocr.jpg')
  ```

- array: 图像的 numpy 数组。它应该是 BGR 格式。

  ```python
  >>> import mmcv
  >>> array = mmcv.imread('demo/demo_text_ocr.jpg')
  >>> inferencer(array)
  ```

- list: 基本类型的列表。列表中的每个元素都将单独处理。

  ```python
  >>> inferencer(['img_1.jpg', 'img_2.jpg])
  >>> # 你甚至可以混合类型
  >>> inferencer(['img_1.jpg', array])
  ```

- str: 目录的路径。目录中的所有图像都将被处理。

  ```python
  >>> inferencer('tests/data/det_toy_dataset/imgs/test/')
  ```

````

````{tab} KIEInferencer

输入可以是一个字典或者一个字典列表，其中每个字典包含以下键：

- `img` (str 或者 ndarray): 图像的路径或图像本身。如果 KIE 推理器在无可视模式下使用，则不需要此键。如果它是一个 numpy 数组，则应该是 BGR 顺序编码的图片。
- `img_shape` (tuple(int, int)): 图像的形状 (H, W)。仅在 KIE 推理器在无可视模式下使用且没有提供 `img` 时才需要。
- `instances` (list[dict]): 实例列表。

每个 `instance` 都应该包含以下键：

```python
{
    # 一个嵌套列表，其中包含 4 个数字，表示实例的边界框，顺序为 (x1, y1, x2, y2)
    "bbox": np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                    dtype=np.int32),

    # 文本列表
    "texts": ['text1', 'text2', ...],
}
```

````
`````

### 输出

默认情况下，每个推理器都以字典格式返回预测结果。

- `visualization` 包含可视化的预测结果。但默认情况下，它是一个空列表，除非 `return_vis=True`。

- `predictions` 包含以 json-可序列化格式返回的预测结果。如下所示，内容因任务类型而异。

  `````{tabs}

  :::group-tab} MMOCRInferencer

  ```python
  {
      'predictions' : [
        # 每个实例都对应于一个输入图像
        {
          'det_polygons': [...],  # 2d 列表，长度为 (N,)，格式为 [x1, y1, x2, y2, ...]
          'det_scores': [...],  # 浮点列表，长度为（N, ）
          'det_bboxes': [...],   # 2d 列表，形状为 (N, 4)，格式为 [min_x, min_y, max_x, max_y]
          'rec_texts': [...],  # 字符串列表，长度为（N, ）
          'rec_scores': [...],  # 浮点列表，长度为（N, ）
          'kie_labels': [...],  # 节点标签，长度为 (N, )
          'kie_scores': [...],  # 节点置信度，长度为 (N, )
          'kie_edge_scores': [...],  # 边预测置信度, 形状为 (N, N)
          'kie_edge_labels': [...]  # 边标签, 形状为 (N, N)
        },
        ...
      ],
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  ```

  :::

  ````{group-tab} 标准推理器

  ::::{tabs}
  :::{code-tab} python TextDetInferencer

  {
      'predictions' : [
        # 每个实例都对应于一个输入图像
        {
          'polygons': [...],  # 2d 列表，长度为 (N,)，格式为 [x1, y1, x2, y2, ...]
          'bboxes': [...],  # 2d 列表，形状为 (N, 4)，格式为 [min_x, min_y, max_x, max_y]
          'scores': [...]  # 浮点列表，长度为（N, ）
        },
        ...
      ]
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  :::

  :::{code-tab} python TextRecInferencer
  {
      'predictions' : [
        # 每个实例都对应于一个输入图像
        {
          'text': '...',  # 字符串
          'scores': 0.1,  # 浮点
        },
        ...
      ]
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  :::

  :::{code-tab} python TextSpottingInferencer
  {
      'predictions' : [
        # 每个实例都对应于一个输入图像
        {
          'polygons': [...],  # 2d 列表，长度为 (N,)，格式为 [x1, y1, x2, y2, ...]
          'bboxes': [...],  # 2d 列表，形状为 (N, 4)，格式为 [min_x, min_y, max_x, max_y]
          'scores': [...]  # 浮点列表，长度为（N, ）
          'texts': ['...',]  # 字符串列表，长度为（N, ）
        },
      ]
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  :::

  :::{code-tab} python KIEInferencer
  {
      'predictions' : [
        # 每个实例都对应于一个输入图像
        {
          'labels': [...],  # 节点标签，长度为 (N, )
          'scores': [...],  # 节点置信度，长度为 (N, )
          'edge_scores': [...],  # 边预测置信度, 形状为 (N, N)
          'edge_labels': [...],  # 边标签, 形状为 (N, N)
        },
      ]
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  :::
  ::::{tabs}
  ````

  `````

如果你想要从模型中获取原始输出，可以将 `return_datasamples` 设置为 `True` 来获取原始的 [DataSample](structures.md)，它将存储在 `predictions` 中。

### 储存结果

除了从返回值中获取预测结果，你还可以通过设置 `out_dir` 和 `save_pred`/`save_vis` 参数将预测结果和可视化结果导出到文件中。

```python
>>> inferencer('img_1.jpg', out_dir='outputs/', save_pred=True, save_vis=True)
```

结果目录结构如下：

```text
outputs
├── preds
│   └── img_1.json
└── vis
    └── img_1.jpg
```

文件名与对应的输入图像文件名相同。 如果输入图像是数组，则文件名将是从0开始的数字。

### 批量推理

你可以通过设置 `batch_size` 来自定义批量推理的批大小。 默认批大小为 1。

## API

这里列出了推理器详尽的参数列表。

````{tabs}

```{group-tab} MMOCRInferencer

**MMOCRInferencer.\_\_init\_\_():**

| 参数          | 类型                                      | 默认值 | 描述                                                                                                                           |
| ------------- | ----------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `det`         | str 或 [权重](../modelzoo.html#id2), 可选 | None   | 预训练的文本检测算法。它是配置文件的路径或者是 metafile 中定义的模型名称。                                                     |
| `det_weights` | str, 可选                                 | None   | det 模型的权重文件的路径。                                                                                                     |
| `rec`         | str 或 [权重](../modelzoo.html#id2), 可选 | None   | 预训练的文本识别算法。它是配置文件的路径或者是 metafile 中定义的模型名称。                                                     |
| `rec_weights` | str, 可选                                 | None   | rec 模型的权重文件的路径。                                                                                                     |
| `kie` \[1\]   | str 或 [权重](../modelzoo.html#id2), 可选 | None   | 预训练的关键信息提取算法。它是配置文件的路径或者是 metafile 中定义的模型名称。                                                 |
| `kie_weights` | str, 可选                                 | None   | kie 模型的权重文件的路径。                                                                                                     |
| `device`      | str, 可选                                 | None   | 推理使用的设备，接受 `torch.device` 允许的所有字符串。例如，'cuda:0' 或 'cpu'。如果为 None，将自动使用可用设备。 默认为 None。 |

\[1\]: 当同时指定了文本检测和识别模型时，`kie` 才会生效。

**MMOCRInferencer.\_\_call\_\_()**

| 参数                 | 类型                    | 默认值     | 描述                                                                                           |
| -------------------- | ----------------------- | ---------- | ---------------------------------------------------------------------------------------------- |
| `inputs`             | str/list/tuple/np.array | **必需**   | 它可以是一个图片/文件夹的路径，一个 numpy 数组，或者是一个包含图片路径或 numpy 数组的列表/元组 |
| `return_datasamples` | bool                    | False      | 是否将结果作为 DataSample 返回。如果为 False，结果将被打包成一个字典。                         |
| `batch_size`         | int                     | 1          | 推理的批大小。                                                                                 |
| `return_vis`         | bool                    | False      | 是否返回可视化结果。                                                                           |
| `print_result`       | bool                    | False      | 是否将推理结果打印到控制台。                                                                   |
| `show`               | bool                    | False      | 是否在弹出窗口中显示可视化结果。                                                               |
| `wait_time`          | float                   | 0          | 弹窗展示可视化结果的时间间隔。                                                                 |
| `out_dir`            | str                     | `results/` | 结果的输出目录。                                                                               |
| `save_vis`           | bool                    | False      | 是否将可视化结果保存到 `out_dir`。                                                             |
| `save_pred`          | bool                    | False      | 是否将推理结果保存到 `out_dir`。                                                               |

```

```{group-tab} 标准推理器

**Inferencer.\_\_init\_\_():**

| 参数      | 类型                                      | 默认值 | 描述                                                                                                                               |
| --------- | ----------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `model`   | str 或 [权重](../modelzoo.html#id2), 可选 | None   | 路径到配置文件或者在 metafile 中定义的模型名称。                                                                                   |
| `weights` | str, 可选                                 | None   | 权重文件的路径。                                                                                                                   |
| `device`  | str, 可选                                 | None   | 推理使用的设备，接受 `torch.device` 允许的所有字符串。 例如，'cuda:0' 或 'cpu'。 如果为 None，则将自动使用可用设备。 默认为 None。 |

**Inferencer.\_\_call\_\_()**

| 参数                 | 类型                    | 默认值     | 描述                                                                                |
| -------------------- | ----------------------- | ---------- | ----------------------------------------------------------------------------------- |
| `inputs`             | str/list/tuple/np.array | **必需**   | 可以是图像的路径/文件夹，np 数组或列表/元组（带有图像路径或 np 数组）               |
| `return_datasamples` | bool                    | False      | 是否将结果作为 DataSamples 返回。 如果为 False，则结果将被打包到一个 dict 中。      |
| `batch_size`         | int                     | 1          | 推理批大小。                                                                        |
| `progress_bar`       | bool                    | True       | 是否显示进度条。                                                                    |
| `return_vis`         | bool                    | False      | 是否返回可视化结果。                                                                |
| `print_result`       | bool                    | False      | 是否将推理结果打印到控制台。                                                        |
| `show`               | bool                    | False      | 是否在弹出窗口中显示可视化结果。                                                    |
| `wait_time`          | float                   | 0          | 弹窗展示可视化结果的时间间隔。                                                      |
| `draw_pred`          | bool                    | True       | 是否绘制预测的边界框。 *仅适用于 `TextDetInferencer` 和 `TextSpottingInferencer`。* |
| `out_dir`            | str                     | `results/` | 结果的输出目录。                                                                    |
| `save_vis`           | bool                    | False      | 是否将可视化结果保存到 `out_dir`。                                                  |
| `save_pred`          | bool                    | False      | 是否将推理结果保存到 `out_dir`。                                                    |

```
````

## 命令行接口

```{note}
该节仅适用于 `MMOCRInferencer`.
```

`MMOCRInferencer` 的命令行形式可以通过 `tools/infer.py` 调用，大致形式如下：

```bash
python tools/infer.py INPUT_PATH [--det DET] [--det-weights ...] ...
```

其中，`INPUT_PATH` 为必须字段，内容应当为指向图片或文件目录的路径。其他参数与 Python 接口基本一致，遵循如下映射关系：

- 你需要在每个参数前面加上两个`-`，然后把下划线`_`替换成连字符`-`。例如， `out_dir` 会变成 `--out-dir`。
- 对于布尔类型的参数，将参数放在命令中就相当于将其指定为 True。例如， `--show` 会将 `show` 参数指定为 True。

此外，命令行中默认不会回显推理结果，你可以通过 `--print-result` 参数来查看推理结果。

下面是一个例子：

```bash
python tools/infer.py demo/demo_text_ocr.jpg --det DBNet --rec SAR --show --print-result
```

可以得到如下结果：

```bash
{'predictions': [{'rec_texts': ['CBank', 'Docbcba', 'GROUP', 'MAUN', 'CROBINSONS', 'AOCOC', '916M3', 'BOO9', 'Oven', 'BRANDS', 'ARETAIL', '14', '70<UKN>S', 'ROUND', 'SALE', 'YEAR', 'ALLY', 'SALE', 'SALE'],
'rec_scores': [0.9753464579582214, ...], 'det_polygons': [[551.9930285844646, 411.9138765335083, 553.6153911653112,
383.53195309638977, 620.2410061195247, 387.33785033226013, 618.6186435386782, 415.71977376937866], ...], 'det_scores': [0.8230461478233337, ...]}]}
```
