# 推理器 （Inferencers）

在 OpenMMLab 中，所有的推理操作都被统一到了新的推理器 - `Inferencer` 中。`Inferencer` 被设计成为一个简洁易用的 API，它在不同的 OpenMMLab 库中都有着非常相似的接口。

在 MMOCR 中，推理器被构建在不同层次的任务抽象中。

- 运行特定任务的推理器：遵循 OpenMMLab 的惯例，MMOCR 中的每个基本任务都有自己的推理器，即 `TextDetInferencer`，`TextRecInferencer`，`TextSpottingInferencer` 和 `KIEInferencer`。它们被设计成用于对单个任务进行推理，并且可以被链接在一起，以便对一系列任务进行推理。它们还具有非常相似的接口，具有标准的输入/输出协议，并且总体遵循 OpenMMLab 的设计。
- [`MMOCRInferencer`](../user_guides/inference.md)：我们还提供了 `MMOCRInferencer`，一个专门为 MMOCR 设计的便捷推理接口。它封装和链接了 MMOCR 中的所有推理器，因此用户可以使用此推理器对图像执行一系列任务，并直接以端到端的方式获得最终结果。*但是，它的接口与其他特定任务的推理器有很大不同，并且为了简单起见，可能会牺牲一些标准的推理器功能。*

对于新用户，我们建议使用 [`MMOCRInferencer`](../user_guides/inference.md) 来测试不同模型的组合。

如果您是开发人员并希望将模型集成到自己的项目中，我们建议使用特定于任务的推理器，因为它们更灵活且标准化，并具有完整的功能。

本页将介绍**特定于任务的推理器**的用法。

## 基础用法

通常，OpenMMLab 中的所有特定于任务的推理器都具有非常相似的接口。下面的例子展示了如何使用 `TextDetInferencer` 对单个图像进行推理。

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

## 初始化

每个推理器必须使用一个模型进行初始化，也可以手动选择推理设备。

### 模型初始化

每个特定于任务的 `Inferencer` 都接受两个参数，`model` 和 `weights`。 （在 `MMOCRInferencer` 中，它们被称为 `xxx` 和 `xxx_weights`）

- `model` 接受模型的名称或配置文件的路径作为输入。模型的名称从 [model-index.yml](https://github.com/open-mmlab/mmocr/blob/1.x/model-index.yml) 中的模型的元文件（[示例](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/metafile.yml) ）中获取。您可以在[此处](../modelzoo.md#weights)找到可用权重的列表。

  ```{note}
  为了方便起见，我们在其元文件的“别名”字段中缩写了一些常用模型的名称，Inferencer 也可以使用这些名称来索引模型。
  ```

- `weights` 接受权重文件的路径。

此处列举了一些常见的初始化模型的方法。

- 你可以通过传递模型的名称给 `model` 来推理 MMOCR 的预训练模型。权重将会自动从 OpenMMLab 的模型库中下载并加载。

  ```python
  >>> from mmocr.apis import TextDetInferencer
  >>> inferencer = TextDetInferencer(model='DBNet')
  ```

  ```{note}
  模型种类必须与推理器种类匹配。
  ```

  你可以通过将权重的路径/URL传递给 `weights` 来让推理器加载你自己的权重。

  ```python
  >>> inferencer = TextDetInferencer(model='DBNet', weights='path/to/dbnet.pth')
  ```

- 如果你有自己的配置和权重，你可以将配置文件的路径传递给 `model`，将权重的路径传递给 `weights`。

  ```python
  >>> inferencer = TextDetInferencer(model='path/to/dbnet_config.py', weights='path/to/dbnet.pth')
  ```

- 默认情况下，[MMEngine](https://github.com/open-mmlab/mmengine/) 会自动将配置文件转储到权重文件中。如果您有一个在 MMEngine 上训练的权重，您也可以将权重文件的路径传递给 `weights`，而不需要指定 `model`：

  ```python
  >>> # 如果无法在权重中找到配置文件，则会引发错误
  >>> inferencer = TextDetInferencer(weights='path/to/dbnet.pth')
  ```

- 传递配置文件到 `model` 而不指定 `weight` 将导致随机初始化的模型。

### 推理设备

每个Inferencer实例都会跟一个设备绑定。默认情况下，最佳设备是由 [MMEngine](https://github.com/open-mmlab/mmengine/) 自动决定的。你也可以通过指定 `device` 参数来改变设备。例如，你可以使用以下代码在 GPU 1上创建一个 Inferencer。

```python
>>> inferencer = TextDetInferencer(model='DBNet', device='cuda:1')
```

如要在 CPU 上创建一个 Inferencer：

```python
>>> inferencer = TextDetInferencer(model='DBNet', device='cpu')
```

请参考 [torch.device](torch.device) 了解 `device` 参数支持的所有形式。

## 推理

当推理器初始化后，你可以直接传入要推理的原始数据，从返回值中获取推理结果。

### 输入

`````{tabs}

````{tab} TextDetInferencer / TextRecInferencer / TextSpottingInferencer

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

Each `instance` looks like the following:

```python
{
    # A nested list of 4 numbers representing the bounding box of
    # the instance, in (x1, y1, x2, y2) order.
    "bbox": np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                    dtype=np.int32),

    # List of texts.
    "texts": ['text1', 'text2', ...],
}
```

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

默认情况下，每个 `Inferencer` 都以字典格式返回预测结果。

- `visualization` 包含可视化的预测结果。但默认情况下，它是一个空列表，除非 `return_vis=True`。

- `predictions` 包含以 json-可序列化格式返回的预测结果。如下所示，内容因任务类型而异。

  `````{tabs}

  ```{code-tab} python TextDetInferencer

  {
      'predictions' : [
        # 每个实例都对应于一个输入图像
        {
          'polygons': [...],  # 2d 列表，长度为 (N,)，格式为 [x1, y1, x2, y2, ...]
          'bboxes': [...],  # 2d 列表，形状为 (N, 4)，格式为 [min_x, min_y, max_x, max_y]
          'scores': [...]  # 浮点列表，长度为（N, ）
        },
  ```

  ```{code-tab} python TextRecInferencer
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
  ```

  ```{code-tab} python TextSpottingInferencer
  {
      'predictions' : [
        # 每个实例都对应于一个输入图像
        {
          'polygons': [...],  # 2d 列表，长度为 (N,)，格式为 [x1, y1, x2, y2, ...]
          'bboxes': [...],  # 2d 列表，形状为 (N, 4)，格式为 [min_x, min_y, max_x, max_y]
          'scores': [...]  # 浮点列表，长度为（N, ）
          'texts': ['...',]  # 浮点列表，长度为（N, ）
        },
      ]
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  ```

  ```{code-tab} python KIEInferencer
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
  ```
  ````{tabs}

  `````

如果你想要从模型中获取原始输出，可以将 `return_datasamples` 设置为 `True` 来获取原始的 [DataSample](structures.md)，它将存储在 `predictions` 中。

### 储存结果

除了从返回值中获取预测结果，您还可以通过设置 `out_dir` 和 `save_pred`/`save_vis` 参数将预测结果/可视化导出到文件中。

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

这里列出了 Inferencer 详尽的参数列表。 除非另有说明，否则它们通常适用于所有 Inferencer。

**Inferencer.\_\_init\_\_():**

| 参数      | 类型                                          | 默认值 | 描述                                                                                                                               |
| --------- | --------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `model`   | str 或 [权重](../modelzoo.html#id2), optional | None   | 路径到配置文件或者在 metafile 中定义的模型名称。                                                                                   |
| `weights` | str, 可选                                     | None   | 权重文件的路径。                                                                                                                   |
| `device`  | str, 可选                                     | None   | 推理使用的设备，接受 `torch.device` 允许的所有字符串。 例如，'cuda:0' 或 'cpu'。 如果为 None，则将自动使用可用设备。 默认为 None。 |

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
