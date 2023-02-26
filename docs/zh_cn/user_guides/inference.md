# 推理

我们提供了一个易于使用的API--"MMOCRInferencer"，它可以对以下任务进行推理。

- 文本检测
- 文本识别
- OCR（文本检测 + 文本识别）
- 关键信息提取（文本检测 + 文本识别 + 关键信息提取）
- *OCR（text spotting）*（即将推出）

这些任务是通过使用一个或几个特定任务的 \[Inferencer\]（.../basic_concepts/inferencers.md）来完成的。`MMOCRInferencer` 封装并链接了MMOCR中的所有 Inferencer，因此用户可以使用它在图像上执行一系列任务，并直接以端到端的方式获得最终结果。

下面的章节将引导你了解 `MMOCRInferencer` 的一些基本使用方法。

## 基本用法

要对 demo/demo_text_ocr.jpg 进行 OCR，使用 `DBNet` 作为文本检测模型，`CRNN` 作为文本识别模型，只需执行以下命令:

```python
>>> from mmocr.apis import MMOCRInferencer
>>> # 读取模型
>>> ocr = MMOCRInferencer(det='DBNet', rec='SAR')
>>> # 进行推理并可视化结果
>>> ocr('demo/demo_text_ocr.jpg', show=True)
```

可视化结果将被显示在一个新窗口中：

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/220563262-e9c1ab52-9b96-4d9c-bcb6-f55ff0b9e1be.png" height="250"/>
</div>

```{note}
如果你在没有GUI的服务器上运行 MMOCR，或者是通过禁用 X11 转发的 SSH 隧道运行该指令，`show`  选项将不起作用。然而，你仍然可以通过设置 `out_dir' 和 `save_vis=True' 参数将可视化数据保存到文件。阅读 [获取结果](#TODO) 了解详情。
```

根据初始化参数，`MMOCRInferencer`可以在不同模式下运行。例如，如果初始化时指定了 `det`、`rec` 和 `kie`，它可以在 KIE 模式下运行。

```python
>>> kie = MMOCRInferencer(det='DBNet', rec='SAR', kie='SDMGR')
>>> kie('demo/demo_kie.jpeg', show=True)
```

可视化结果如下：

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/220569700-fd4894bc-f65a-405e-95e7-ebd2d614aedd.png" height="250"/>
</div>
<br />

`MMOCRInferencer` 接受各种类型的输入。它可以是一个 numpy 数组或图片的路径/url。如果你有多个输入，也可以将他们包在一个列表内传入。

```python
>>> import mmcv
>>> # 将图片以 numpy 数组的方式读入
>>> np_img = mmcv.imread('tests/data/det_toy_dataset/imgs/test/img_1.jpg')
>>> # 传入一个列表，混合字符串和 numpy 数组的方式也是可以接受的
>>> ocr([np_img,  'tests/data/det_toy_dataset/imgs/test/img_10.jpg'], show=True)
```

如果你需要遍历一个目录下的所有图片文件，只需要将路径以字符串的方式传入 `MMOCRInferencer` 即可：

```python
>>> ocr('tests/data/det_toy_dataset/imgs/test/', show=True)
```

## 模型初始化

对于每个任务，`MMOCRInferencer`需要两个参数`xxx` 和 `xxx_weights` （例如`det`和`det_weights`）以对模型进行初始化。我们有很多方法来初始化推理的模型。此处将以`det`和`det_weights`为例来说明一些典型的初始化模型的方法。

- 要用 MMOCR 的预训练模型进行推理，只需要把它的名字传给参数 `det`，权重将自动从 OpenMMLab 的模型库中下载和加载。[此处](../modelzoo.md#id2)记录了 MMOCR 中可以通过该方法初始化的所有模型。

  ```python
  >>> MMOCRInferencer(det='DBNet')
  ```

- 要加载自定义的配置和权重，你可以把配置文件的路径传给 `det`，把权重的路径传给 `det_weights`。

  ```python
  >>> MMOCRInferencer(det='path/to/dbnet_config.py', det_weights='path/to/dbnet.pth')
  ```

[这里](../basic_concepts/inferencers.md#model-initialization)还列出了更多种初始化 `Inferencer` 的方式。

## 推理设备

每个Inferencer实例都会跟一个设备绑定。默认情况下，最佳设备是由 [MMEngine](https://github.com/open-mmlab/mmengine/) 自动决定的。你也可以通过指定 `device` 参数来改变设备。请参考 [torch.device](torch.device) 了解 `device` 参数支持的所有形式。

## 批量推理

你可以通过设置 `batch_size` 来自定义批大小。默认的批大小为1。

## 获取结果

在 Python 接口中，`MMOCRInferencer` 以字典的形式返回预测结果。字典的键以任务名开头，例如 `det`、`rec` 和 `kie`；字典的值是对应的预测结果。具体返回的结果可能是以下内容的一个子集，取决于 `MMOCRInferencer` 正在运行的任务。

```python
{
    'predictions' : [{
        'det_polygons': [...],
        'det_scores': [...]
        'rec_texts': [...],
        'rec_scores': [...],
        'kie_labels': [...],
        'kie_scores': [...],
        'kie_edge_scores': [...],
        'kie_edge_labels': [...]
    },]
    'visualization' : [array(..., dtype=uint8),]
}
```

`predictions` 中的每个字典都包含了对应图片的推理结果。`visualization` 中的每个 numpy 数组都对应了一个图片的可视化结果。

```{note}
可视化结果只有在 `return_vis=True` 时才会被返回。
```

除了从返回值中获取预测结果，你还可以通过设置 `out_dir` 和 `save_pred`/`save_vis` 参数将预测结果及可视化结果导出到文件中。假设 `out_dir` 是 `outputs`，文件将会按如下方式组织：

```text
outputs
├── preds
│   └── img_1.json
└── vis
    └── img_1.jpg
```

每个文件的文件名与对应的输入图片文件名相同。如果输入图片是 numpy 数组，文件名将从 0 开始。

## CLI

`MMOCRInferencer` supports both CLI and Python interface. All arguments are the same for the CLI, all you need to do is add 2 hyphens at the beginning of the argument and replace underscores by hyphens.
(*Example:* `out_dir` becomes `--out-dir`)

`MMOCRInferencer` 可以通过 CLI 和 Python 接口使用。CLI 和 Python 接口的所有参数都是一样的，你只需要在参数前面加上两个"-"，然后把下划线"\_"替换成连字符"-"即可。(*例如:* `out_dir` 变成 `--out-dir`)

对于布尔类型的参数，将参数放在命令中就相当于将其存储为 true。

例如，[第一个例子](#基本用法)可以在 CLI 中这样运行：

```bash
python tools/infer.py demo/demo_text_ocr.jpg --det DBNet --rec CRNN --show
```

## API

该 API 有多个可供使用的参数列表。下表是 python 接口的参数。

**MMOCRInferencer.\_\_init\_\_():**

| 参数          | 类型                                             | 默认值 | 描述                                                                                                                           |
| ------------- | ------------------------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `det`         | str 或 [权重](../modelzoo.html#weights), 可选    | None   | 预训练的文本检测算法。它是配置文件的路径或者是 metafile 中定义的模型名称。                                                     |
| `det_weights` | str, 可选                                        | None   | det 模型的权重文件的路径。如果它没有被指定，并且 "det" 是 metafile 中的模型名称，那么权重将从 metafile 中加载。                |
| `rec`         | str 或 [Weights](../modelzoo.html#weights), 可选 | None   | 预训练的文本识别算法。它是配置文件的路径或者是 metafile 中定义的模型名称。                                                     |
| `rec_weights` | str, 可选                                        | None   | rec 模型的权重文件的路径。如果它没有被指定，并且 "rec" 是 metafile 中的模型名称，那么权重将从 metafile 中加载。                |
| `kie` \[1\]   | str 或 [Weights](../modelzoo.html#weights), 可选 | None   | 预训练的关键信息提取算法。它是配置文件的路径或者是 metafile 中定义的模型名称。                                                 |
| `kie_weights` | str, 可选                                        | None   | kie 模型的权重文件的路径。如果它没有被指定，并且 "kie" 是 metafile 中的模型名称，那么权重将从 metafile 中加载。                |
| `device`      | str, 可选                                        | None   | 推理使用的设备，接受 `torch.device` 允许的所有字符串。例如，'cuda:0' 或 'cpu'。如果为 None，将自动使用可用设备。 默认为 None。 |

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
