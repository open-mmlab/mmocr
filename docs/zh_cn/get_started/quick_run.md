# 快速运行

这个章节会介绍 MMOCR 的一些基本功能。我们假设你已经[从源码安装了 MMOCR](install.md#best-practices)。

## 推理

在 MMOCR 的根目录下运行以下命令：

```shell
python tools/infer.py demo/demo_text_ocr.jpg --det DBNet --rec CRNN --show --print-result
```

你可以看到弹出的预测结果，以及在控制台中打印出的推理结果。

<div align="center">
    <img src="https://user-images.githubusercontent.com/24622904/187825445-d30cbfa6-5549-4358-97fe-245f08f4ed94.jpg" height="250"/>
</div>
<br/ >

```bash
# 识别结果
{'predictions': [{'rec_texts': ['cbanks', 'docecea', 'grouf', 'pwate', 'chobnsonsg', 'soxee', 'oeioh', 'c', 'sones', 'lbrandec', 'sretalg', '11', 'to8', 'round', 'sale', 'year',
'ally', 'sie', 'sall'], 'rec_scores': [...], 'det_polygons': [...], 'det_scores':
[...]}]}
```

```{note}
如果你在没有 GUI 的服务器上运行 MMOCR，或者通过没有开启 X11 转发的 SSH 隧道运行 MMOCR，你可能无法看到弹出的窗口。
```

对 MMOCR 中推理接口更为详细的说明，可以在[这里](../user_guides/inference.md)找到。

除了使用我们提供好的预训练模型，用户也可以在自己的数据集上训练流行模型。接下来我们以在迷你的 [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) 数据集上训练 DBNet 为例，带大家熟悉 MMOCR 的基本功能。

## 准备数据集

由于 OCR 任务的数据集种类多样，格式不一，不利于多数据集的切换和联合训练，因此 MMOCR 约定了一种[统一的数据格式](../user_guides/dataset_prepare.md)，并针对常用的 OCR 数据集提供了[一键式数据准备脚本](../user_guides/data_prepare/dataset_preparer.md)。通常，要在 MMOCR 中使用数据集，你只需要按照对应步骤运行指令即可。

```{note}
但我们亦深知，效率就是生命——尤其对想要快速上手 MMOCR 的你来说。
```

在这里，我们准备了一个用于演示的精简版 ICDAR 2015 数据集。下载我们预先准备好的[压缩包](https://download.openmmlab.com/mmocr/data/icdar2015/mini_icdar2015.tar.gz)，解压到 mmocr 的 `data/` 目录下，就能得到我们准备好的图片和标注文件。

```Bash
wget https://download.openmmlab.com/mmocr/data/icdar2015/mini_icdar2015.tar.gz
mkdir -p data/
tar xzvf mini_icdar2015.tar.gz -C data/
```

## 修改配置

准备好数据集后，我们接下来就需要通过修改配置的方式指定训练集的位置和训练参数。

在这个例子中，我们将会训练一个以 resnet18 作为骨干网络（backbone）的 DBNet。由于 MMOCR 已经有针对完整 ICDAR 2015 数据集的配置 （`configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py`），我们只需要在它的基础上作出一点修改。

我们首先需要修改数据集的路径。在这个配置中，大部分关键的配置文件都在 `_base_` 中被导入，如数据库的配置就来自 `configs/textdet/_base_/datasets/icdar2015.py`。打开该文件，把第一行 `icdar2015_textdet_data_root` 指向的路径替换：

```Python
icdar2015_textdet_data_root = 'data/mini_icdar2015'
```

另外，因为数据集尺寸缩小了，我们也要相应地减少训练的轮次到 400，缩短验证和储存权重的间隔到10轮，并放弃学习率衰减策略。直接把以下几行配置放入 `configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py`即可生效：

```Python
# 每 10 个 epoch 储存一次权重，且只保留最后一个权重
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=1,
    ))
# 设置最大 epoch 数为 400，每 10 个 epoch 运行一次验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400, val_interval=10)
# 令学习率为常量，即不进行学习率衰减
param_scheduler = [dict(type='ConstantLR', factor=1.0),]
```

这里，我们通过配置的继承 ({external+mmengine:doc}`MMEngine: Config <advanced_tutorials/config>`) 机制将基础配置中的相应参数直接进行了改写。原本的字段分布在 `configs/textdet/_base_/schedules/schedule_sgd_1200e.py` 和 `configs/textdet/_base_/default_runtime.py` 中，感兴趣的读者可以自行查看。

```{note}
关于配置文件更加详尽的说明，请参考[此处](../user_guides/config.md)。
```

## 可视化数据集

在正式开始训练前，我们还可以可视化一下经过训练过程中[数据变换（transforms）](../basic_concepts/transforms.md)后的图像。方法也很简单，把我们需要可视化的配置传入 [browse_dataset.py](/tools/analysis_tools/browse_dataset.py) 脚本即可：

```Bash
python tools/analysis_tools/browse_dataset.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py
```

数据变换后的图片和标签会在弹窗中逐张被展示出来。

<center class="half">
    <img src="https://user-images.githubusercontent.com/24622904/187611542-01e9aa94-fc12-4756-964b-a0e472522a3a.jpg" width="250"/><img src="https://user-images.githubusercontent.com/24622904/187611555-3f5ea616-863d-4538-884f-bccbebc2f7e7.jpg" width="250"/><img src="https://user-images.githubusercontent.com/24622904/187611581-88be3970-fbfe-4f62-8cdf-7a8a7786af29.jpg" width="250"/>
</center>

```{note}
有关该脚本更详细的指南，请参考[此处](../user_guides/useful_tools.md).
```

```{tip}
除了满足好奇心之外，可视化还可以帮助我们在训练前检查可能影响到模型表现的部分，如配置文件、数据集及数据变换中的问题。
```

## 训练

万事俱备，只欠东风。运行以下命令启动训练：

```Bash
python tools/train.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py
```

根据系统情况，MMOCR 会自动使用最佳的设备进行训练。如果有 GPU，则会默认在第一张卡启动单卡训练。当开始看到 loss 的输出，就说明你已经成功启动了训练。

```Bash
2022/08/22 18:42:22 - mmengine - INFO - Epoch(train) [1][5/7]  lr: 7.0000e-03  memory: 7730  data_time: 0.4496  loss_prob: 14.6061  loss_thr: 2.2904  loss_db: 0.9879  loss: 17.8843  time: 1.8666
2022/08/22 18:42:24 - mmengine - INFO - Exp name: dbnet_resnet18_fpnc_1200e_icdar2015
2022/08/22 18:42:28 - mmengine - INFO - Epoch(train) [2][5/7]  lr: 7.0000e-03  memory: 6695  data_time: 0.2052  loss_prob: 6.7840  loss_thr: 1.4114  loss_db: 0.9855  loss: 9.1809  time: 0.7506
2022/08/22 18:42:29 - mmengine - INFO - Exp name: dbnet_resnet18_fpnc_1200e_icdar2015
2022/08/22 18:42:33 - mmengine - INFO - Epoch(train) [3][5/7]  lr: 7.0000e-03  memory: 6690  data_time: 0.2101  loss_prob: 3.0700  loss_thr: 1.1800  loss_db: 0.9967  loss: 5.2468  time: 0.6244
2022/08/22 18:42:33 - mmengine - INFO - Exp name: dbnet_resnet18_fpnc_1200e_icdar2015
```

在不指定额外参数时，训练的权重默认会被保存到 `work_dirs/dbnet_resnet18_fpnc_1200e_icdar2015/` 下面，而日志则会保存在`work_dirs/dbnet_resnet18_fpnc_1200e_icdar2015/开始训练的时间戳/`里。接下来，我们只需要耐心等待模型训练完成即可。

```{note}
若需要了解训练的高级用法，如 CPU 训练、多卡训练及集群训练等，请查阅[训练与测试](../user_guides/train_test.md)。
```

## 测试

经过数十分钟的等待，模型顺利完成了400 epochs的训练。我们通过控制台的输出，观察到 DBNet 在最后一个 epoch 的表现最好，`hmean` 达到了 60.86（你可能会得到一个不太一样的结果）：

```Bash
08/22 19:24:52 - mmengine - INFO - Epoch(val) [400][100/100]  icdar/precision: 0.7285  icdar/recall: 0.5226  icdar/hmean: 0.6086
```

```{note}
它或许还没被训练到最优状态，但对于一个演示而言已经足够了。
```

然而，这个数值只反映了 DBNet 在迷你 ICDAR 2015 数据集上的性能。要想更加客观地评判它的检测能力，我们还要看看它在分布外数据集上的表现。例如，`tests/data/det_toy_dataset` 就是一个很小的真实数据集，我们可以用它来验证一下 DBNet 的实际性能。

在测试前，我们同样需要对数据集的位置做一下修改。打开 `configs/textdet/_base_/datasets/icdar2015.py`，修改 `icdar2015_textdet_test` 的 `data_root` 为 `tests/data/det_toy_dataset`:

```Python
# ...
icdar2015_textdet_test = dict(
    type='OCRDataset',
    data_root='tests/data/det_toy_dataset',
    # ...
    )
```

修改完毕，运行命令启动测试。

```Bash
python tools/test.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py work_dirs/dbnet_resnet18_fpnc_1200e_icdar2015/epoch_400.pth
```

得到输出：

```Bash
08/21 21:45:59 - mmengine - INFO - Epoch(test) [5/10]    memory: 8562
08/21 21:45:59 - mmengine - INFO - Epoch(test) [10/10]    eta: 0:00:00  time: 0.4893  data_time: 0.0191  memory: 283
08/21 21:45:59 - mmengine - INFO - Evaluating hmean-iou...
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.30, recall: 0.6190, precision: 0.4815, hmean: 0.5417
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.40, recall: 0.6190, precision: 0.5909, hmean: 0.6047
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.50, recall: 0.6190, precision: 0.6842, hmean: 0.6500
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.60, recall: 0.6190, precision: 0.7222, hmean: 0.6667
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.70, recall: 0.3810, precision: 0.8889, hmean: 0.5333
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.80, recall: 0.0000, precision: 0.0000, hmean: 0.0000
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.90, recall: 0.0000, precision: 0.0000, hmean: 0.0000
08/21 21:45:59 - mmengine - INFO - Epoch(test) [10/10]  icdar/precision: 0.7222  icdar/recall: 0.6190  icdar/hmean: 0.6667
```

可以发现，模型在这个数据集上能达到的 hmean 为 0.6667，效果还是不错的。

```{note}
若需要了解测试的高级用法，如 CPU 测试、多卡测试及集群测试等，请查阅[训练与测试](../user_guides/train_test.md)。
```

## 可视化输出

为了对模型的输出有一个更直观的感受，我们还可以直接可视化它的预测输出。在 `test.py` 中，用户可以通过 `show` 参数打开弹窗可视化；也可以通过 `show-dir` 参数指定预测结果图导出的目录。

```Bash
python tools/test.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py work_dirs/dbnet_resnet18_fpnc_1200e_icdar2015/epoch_400.pth --show-dir imgs/
```

真实标签和预测值会在可视化结果中以平铺的方式展示。左图的绿框表示真实标签，右图的红框表示预测值。

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/187423562-6a85e209-4b12-46ee-8a41-5c67b1ba83f9.png"/><br>
</div>

```{note}
有关更多可视化功能的介绍，请参阅[这里](../user_guides/visualization.md)。
```
