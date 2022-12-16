# 可视化

阅读本文前建议先阅读 MMEngine 的[可视化 (Visualization)](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/visualization.md)文档以初步了解 Visualizer 的定义及相关用法。

简单来说，MMEngine 中实现了用于满足日常可视化需求的可视化器件 [`Visualizer`](mmengine.visualization.Visualizer)，其主要包含三个功能：

- 实现了常用的绘图 API，例如 [`draw_bboxes`](mmengine.visualization.Visualizer.draw_bboxes) 实现了边界盒的绘制功能，[`draw_lines`](mmengine.visualization.Visualizer.draw_lines) 实现了线条的绘制功能。
- 支持将可视化结果、学习率曲线、损失函数曲线以及验证精度曲线等写入多种后端中，包括本地磁盘以及常用的深度学习训练日志记录工具，如 [TensorBoard](https://www.tensorflow.org/tensorboard) 和 [WandB](https://wandb.ai/site)。
- 支持在代码中的任意位置进行调用，例如在训练或测试过程中可视化或记录模型的中间状态，如特征图及验证结果等。

基于 MMEngine 的 Visualizer，MMOCR 内预置了多种可视化工具，用户仅需简单修改配置文件即可使用：

- `tools/analysis_tools/browse_dataset.py` 脚本提供了数据集可视化功能，其可以绘制经过数据变换（Data Transforms）之后的图像及对应的标注内容，详见 [`browse_dataset.py`](useful_tools.md)。
- MMEngine 中实现了 `LoggerHook`，该 Hook 利用 `Visualizer` 将学习率、损失以及评估结果等数据写入  `Visualizer` 设置的后端中，因此通过修改配置文件中的 `Visualizer` 后端，比如修改为`TensorBoardVISBackend` 或 `WandbVISBackend`，可以实现将日志到 `TensorBoard` 或 `WandB` 等常见的训练日志记录工具中，从而方便用户使用这些可视化工具来分析和监控训练流程。
- MMOCR 中实现了`VisualizerHook`，该 Hook 利用 `Visualizer` 将验证阶段或预测阶段的预测结果进行可视化或储存至 `Visualizer` 设置的后端中，因此通过修改配置文件中的 `Visualizer` 后端，比如修改为`TensorBoardVISBackend` 或 `WandbVISBackend`，可以实现将预测的图像存储到 `TensorBoard` 或 `Wandb`中。

## 配置

得益于注册机制的使用，在 MMOCR 中，我们可以通过修改配置文件来设置可视化器件 `Visualizer` 的行为。通常，我们在 `task/_base_/default_runtime.py` 中定义可视化相关的默认配置， 详见[配置教程](config.md)。

```Python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextxxxLocalVisualizer',  # 不同任务使用不同的可视化器
    vis_backends=vis_backends,
    name='visualizer')
```

依据以上示例，我们可以看出 `Visualizer` 的配置主要由两个部分组成，即，`Visualizer`的类型以及其采用的可视化后端 `vis_backends`。

- 针对不同的 OCR 任务，MMOCR 中预置了多种可视化器件，包括 [`TextDetLocalVisualizer`](mmocr.visualization.TextDetLocalVisualizer)，[`TextRecogLocalVisualizer`](mmocr.visualization.TextRecogLocalVisualizer)，[`TextSpottingLocalVisualizer`](mmocr.visualization.TextSpottingLocalVisualizer) 以及[`KIELocalVisualizer`](mmocr.visualization.KIELocalVisualizer)。这些可视化器件依照自身任务的特点对基础的 Visulizer API 进行了拓展，并实现了相应的标签信息接口 `add_datasamples`。例如，用户可以直接使用 `TextDetLocalVisualizer` 来可视化文本检测任务的标签或预测结果。
- MMOCR 默认将可视化后端 `vis_backend` 设置为本地可视化后端 `LocalVisBackend`，将所有可视化结果及其他训练信息保存在本地文件夹中。

## 存储

MMOCR 默认使用本地可视化后端 [`LocalVisBackend`](mmengine.visualization.LocalVisBackend)，`VisualizerHook` 和`LoggerHook` 中存储的模型损失、学习率、模型评估精度以及可视化结果等信息将被默认保存至`{work_dir}/{config_name}/{time}/{vis_data}` 文件夹。此外，MMOCR 也支持其它常用的可视化后端，如 `TensorboardVisBackend` 以及 `WandbVisBackend`用户只需要将配置文件中的 `vis_backends` 类型修改为对应的可视化后端即可。例如，用户只需要在配置文件中插入以下代码块，即可将数据存储至 `TensorBoard` 以及 `WandB`中。

```Python
_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend'),]
```

## 绘制

### 绘制预测结果信息

MMOCR 主要利用 [`VisualizationHook`](mmocr.engine.hooks.VisualizationHook)validation 和 test 的预测结果, 默认情况下 `VisualizationHook`为关闭状态，默认配置如下：

```Python
visualization=dict( # 用户可视化 validation 和 test 的结果
    type='VisualizationHook',
    enable=False,
    interval=1,
    show=False,
    draw_gt=False,
    draw_pred=False)
```

下表为 `VisualizationHook` 支持的参数：

|   参数    |                                        说明                                         |
| :-------: | :---------------------------------------------------------------------------------: |
|  enable   |          VisualizationHook 的开启和关闭由参数enable控制默认是关闭的状态，           |
| interval  | 在VisualizationHook开启的情况下,用以控制多少iteration 存储或展示 val 或 test 的结果 |
|   show    |                          控制是否可视化 val 或 test 的结果                          |
|  draw_gt  |                         val 或 test 的结果是否绘制标注信息                          |
| draw_pred |                         val 或 test 的结果是否绘制预测结果                          |

如果在训练或者测试过程中想开启 `VisualizationHook` 相关功能和配置，仅需修改配置即可，以 `dbnet_resnet18_fpnc_1200e_icdar2015.py`为例， 同时绘制标注和预测，并且将图像展示，配置可进行如下修改

```Python
visualization = _base_.default_hooks.visualization
visualization.update(
    dict(enable=True, show=True, draw_gt=True, draw_pred=True))
```

<div align=center>
<img src="https://user-images.githubusercontent.com/24622904/187426573-8448c827-1336-4416-aebc-e7fccce362cd.png" height="200"/>
</div>

如果只想查看预测结果信息可以只让`draw_pred=True`

```Python
visualization = _base_.default_hooks.visualization
visualization.update(
    dict(enable=True, show=True, draw_gt=False, draw_pred=True))
```

<div align=center>
<img src="https://user-images.githubusercontent.com/24622904/187428385-e6a23120-6445-4c55-a265-c550da692087.png" height="300"/>
</div>

在 `test.py` 过程中进一步简化，提供了 `--show` 和 `--show-dir`两个参数，无需修改配置即可视化测试过程中绘制标注和预测结果。

```Shell
# 展示test 结果
python tools/test.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py dbnet_r18_fpnc_1200e_icdar2015/epoch_400.pth --show

# 指定预测结果的存储位置
python tools/test.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py dbnet_r18_fpnc_1200e_icdar2015/epoch_400.pth --show-dir imgs/
```

<div align=center>
<img src="https://user-images.githubusercontent.com/24622904/187426573-8448c827-1336-4416-aebc-e7fccce362cd.png" height="200"/>
</div>
