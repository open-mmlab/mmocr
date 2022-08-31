# 配置文档

MMOCR 主要使用 Python 文件作为配置文件。其配置文件系统的设计整合了模块化与继承的思想，方便用户进行各种实验。

## 常见用法

```{note}
本小节建议结合 [配置(Config)](https://github.com/open-mmlab/mmengine/blob/5389c115025303400aa26bfa412f6bb796b932ff/docs/zh_cn/tutorials/config.md) 中的初级用法共同阅读。
```

MMOCR 最常用的操作为三种：配置文件的继承，对 `_base_` 变量的引用以及对 `_base_` 变量的修改。对于 `_base_` 的继承与修改, MMEngine.Config 提供了两种语法，一种是针对 Python，Json， Yaml 均可使用的操作；另一种则仅适用于 Python 配置文件。在 MMOCR 中，我们**更推荐使用只针对Python的语法**，因此下文将以此为基础作进一步介绍。

这里以 `configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py` 为例，说明常用的三种用法。

```Python
_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# dataset settings
ic15_det_train = _base_.ic15_det_train
ic15_det_train.pipeline = _base_.train_pipeline
ic15_det_test = _base_.ic15_det_test
ic15_det_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_det_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ic15_det_test)
```

### 配置文件的继承

配置文件存在继承的机制，即一个配置文件 A 可以将另一个配置文件 B 作为自己的基础并直接继承其中的所有字段，从而避免了大量的复制粘贴。

在 dbnet_resnet18_fpnc_1200e_icdar2015.py 中可以看到：

```Python
_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]
```

上述语句会读取列表中的所有基础配置文件，它们中的所有字段都会被载入到 dbnet_resnet18_fpnc_1200e_icdar2015.py 中。我们可以通过在 Python 解释中运行以下语句，了解配置文件被解析后的结构：

```Python
from mmengine import Config
db_config = Config.fromfile('configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py')
print(db_config)
```

可以发现，被解析的配置包含了所有base配置中的字段和信息。

```{note}
请注意：各 _base_ 配置文件中不能存在同名变量。
```

### `_base_` 变量的引用

有时，我们可能需要直接引用 `_base_` 配置中的某些字段，以避免重复定义。假设我们想要获取 `_base_` 配置中的变量 `pseudo`，就可以直接通过 `_base_.pseudo` 获得 `_base_` 配置中的变量。

该语法已广泛用于 MMOCR 的配置中。MMOCR 中各个模型的数据集和管道（pipeline）配置都引用于*基本*配置。如在

```Python
ic15_det_train = _base_.ic15_det_train
# ...
train_dataloader = dict(
    # ...
    dataset=ic15_det_train)
```

### `_base_` 变量的修改

在 MMOCR 中不同算法在不同数据集通常有不同的数据流水线（pipeline)，因此经常会会存在修改数据集中 `pipeline` 的场景。同时还存在很多场景需要修改 `_base_` 配置中的变量，例如想修改某个算法的训练策略，某个模型的某些算法模块（更换 backbone 等）。用户可以直接利用 Python 的语法直接修改引用的 `_base_` 变量。针对 dict，我们也提供了与类属性修改类似的方法，可以直接修改类属性修改字典内的内容。

1. 字典

   这里以修改数据集中的 `pipeline` 为例：

   可以利用 Python 语法修改字典：

   ```python
   # 获取 _base_ 中的数据集
   ic15_det_train = _base_.ic15_det_train
   # 可以直接利用 Python 的 update 修改变量
   ic15_det_train.update(pipeline=_base_.train_pipeline)
   ```

   也可以使用类属性的方法进行修改：

   ```Python
   # 获取 _base_ 中的数据集
   ic15_det_train = _base_.ic15_det_train
   # 类属性方法修改
   ic15_det_train.pipeline = _base_.train_pipeline
   ```

2. 列表

   假设 `_base_` 配置中的变量 `pseudo = [1, 2, 3]`， 需要修改为 `[1, 2, 4]`:

   ```Python
   # pseudo.py
   pseudo = [1, 2, 3]
   ```

   可以直接重写：

   ```Python
   _base_ = ['pseudo.py']
   pseudo = [1, 2, 4]
   ```

   或者利用 Python 语法修改列表：

   ```Python
   _base_ = ['pseudo.py']
   pseudo = _base_.pseudo
   pseudo[2] = 4
   ```

### 命令行修改配置

有时候我们只希望修部分配置，而不想修改配置文件本身。例如实验过程中想更换学习率，但是又不想重新写一个配置文件，可以通过命令行传入参数来覆盖相关配置。

我们可以在命令行里传入 `--cfg-options`，并在其之后的参数直接修改对应字段，例如我们想在运行 train 的时候修改学习率，只需要在命令行执行：

```Shell
python tools/train.py example.py --cfg-options optim_wrapper.optimizer.lr=1
```

更多详细用法参考[命令行修改配置](https://github.com/open-mmlab/mmengine/blob/5389c115025303400aa26bfa412f6bb796b932ff/docs/zh_cn/tutorials/config.md#%E5%91%BD%E4%BB%A4%E8%A1%8C%E4%BF%AE%E6%94%B9%E9%85%8D%E7%BD%AE)

## 配置内容

通过配置文件与注册器的配合，MMOCR 可以在不侵入代码的前提下修改训练参数以及模型配置。具体而言，用户可以在配置文件中对如下模块进行自定义修改：环境配置、Hook 配置、日志配置、训练策略配置、数据相关配置、模型相关配置、评测配置、可视化配置。

本文档将以文字检测算法 `DBNet` 和文字识别算法 `CRNN` 为例来详细介绍 Config 中的内容。

<div id="env_config"></div>

### 环境配置

```Python
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
random_cfg = dict(seed=None)
```

主要包含三个部分：

- 设置所有注册器的默认 `scope` 为 `mmocr`， 保证所有的模块首先从 `MMOCR` 代码库中进行搜索。若果该模块不存在，则继续从上游算法库 `MMEngine` 和 `MMCV` 中进行搜索（详见[注册器](#https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/registry.md)）。

- `env_cfg` 设置分布式环境配置， 更多配置可以详见 [MMEngine Runner](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/runner.md)

- `random_cfg` 设置 numpy， torch，cudnn 等随机种子，更多配置详见 [Runner](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/runner.md)

<div id="hook_config"></div>

### Hook 配置

Hook 主要分为两个部分，默认 hook 以及自定义 hook。默认 hook 为所有任务想要运行所必须的配置，自定义 hook 一般服务于特定的算法或某些特定任务（目前为止 MMOCR 中没有自定义的 Hook）。

```Python
default_hooks = dict(
    timer=dict(type='IterTimerHook'), # 时间记录，包括数据增强时间以及模型推理时间
    logger=dict(type='LoggerHook', interval=1), # 日志打印间隔
    param_scheduler=dict(type='ParamSchedulerHook'), # 与param_scheduler 更新学习率等超参
    checkpoint=dict(type='CheckpointHook', interval=1),# 保存 checkpoint， interval控制保存间隔
    sampler_seed=dict(type='DistSamplerSeedHook'), # 多机情况下设置种子
    sync_buffer=dict(type='SyncBuffersHook'), # 同步多卡情况下，buffer
    visualization=dict( # 用户可视化val 和 test 的结果
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False))
 custom_hooks = []
```

这里简单介绍几个经常可能会变动的 hook，通用的修改方法参考[修改配置](#base-变量的修改)。

- `LoggerHook`：用于配置日志记录器的行为。例如，通过修改 `interval` 可以控制日志打印的间隔，每 `interval` 次迭代 (iteration) 打印一次日志，更多设置可参考 [LoggerHook API](mmengine.hooks.LoggerHook)。

- `CheckpointHook`：用于配置模型断点保存相关的行为，如保存最优权重，保存最新权重等。同样可以修改 `interval` 控制保存 checkpoint 的间隔。更多设置可参考 [CheckpointHook API](mmengine.hooks.CheckpointHook)

- `VisualizationHook`：用于配置可视化相关行为，例如在验证或测试时可视化预测结果，默认为关。同时该 Hook 依赖[可视化配置](#TODO)。想要了解详细功能可以参考 [Visualizer](visualization.md)。更多配置可以参考 [VisualizationHook API](mmocr.engine.hooks.VisualizationHook)。

如果想进一步了解默认 hook 的配置以及功能，可以参考[钩子(Hook)](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/hook.md)。

<div id="log_config"></div>

### 日志配置

此部分主要用来配置日志配置等级以及日志处理器。

```Python
log_level = 'INFO' # 日志记录等级
log_processor = dict(type='LogProcessor',
                        window_size=10,
                        by_epoch=True)
```

- 日志配置等级与 [logging](https://docs.python.org/3/library/logging.html) 的配置一致，

- 日志处理器主要用来控制输出的格式，详细功能可参考[记录日志](https://github.com/open-mmlab/mmengine/blob/f0d8d3f5d92bc3a1f948bc025e3c50f7958b180e/docs/zh_cn/tutorials/logging.md)：

  - `by_epoch=True` 表示按照epoch输出日志，日志格式需要和 `train_cfg` 中的 `type='EpochBasedTrainLoop'` 参数保持一致。例如想按迭代次数输出日志，就需要令  `log_processor` 中的 ` by_epoch=False` 的同时 `train_cfg` 中的 `type = 'IterBasedTrainLoop'`。

  - `window_size` 表示损失的平滑窗口，即最近 `window_size` 次迭代的各种损失的均值。logger 中最终打印的 loss 值为经过各种损失的平均值。

<div id="schedule_config"></div>

### 训练策略配置

此部分主要包含优化器设置、学习率策略和 `Loop` 设置。

对不同算法任务(文字检测，文字识别，关键信息提取)，通常有自己任务常用的调参策略。这里列出了文字识别中的 `CRNN` 所用涉及的相应配置。

```Python
# 优化器
optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adadelta', lr=1.0))
param_scheduler = [dict(type='ConstantLR', factor=1.0)]
train_cfg = dict(type='EpochBasedTrainLoop',
                    max_epochs=5, # 训练轮数
                    val_interval=1) # 评测间隔
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

- `optim_wrapper` : 主要包含两个部分，优化器封装 (OptimWrapper) 以及优化器 (Optimizer)。详情使用信息可见 [MMEngine 优化器封装](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md)

  - 优化器封装支持不同的训练策略，包括混合精度训练(AMP)、梯度累加和梯度截断。

  - 优化器设置中支持了 PyTorch 所有的优化器，所有支持的优化器见 [PyTorch 优化器列表](torch.optim.algorithms)。

- `param_scheduler` : 学习率调整策略，支持大部分 PyTorch 中的学习率调度器，例如 `ExponentialLR`，`LinearLR`，`StepLR`，`MultiStepLR` 等，使用方式也基本一致，所有支持的调度器见[调度器接口文档](mmengine.optim.scheduler), 更多功能可以[参考优化器参数调整策略](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/param_scheduler.md)

- `train/test/val_cfg` : 任务的执行流程，MMEngine 提供了四种流程：`EpochBasedTrainLoop`, `IterBasedTrainLoop`, `ValLoop`, `TestLoop` 更多可以参考[循环控制器](https://github.com/open-mmlab/mmengine/blob/5e1ef1dd6cfd7b8ecbf051b052504b8c35a05b04/docs/zh_cn/design/runner.md#%E5%BE%AA%E7%8E%AF%E6%8E%A7%E5%88%B6%E5%99%A8)。

### 数据相关配置

<div id="dataset_config"></id>

#### 数据集配置

主要用于配置两个方向：

- 数据集的图像与标注文件的位置。

- 数据增强相关的配置。在 OCR 领域中，数据增强通常与模型强相关。

更多参数配置可以参考[数据基类](#TODO)。

数据集字段的命名规则在 MMOCR 中为：

```Python
{数据集名称缩写}_{算法任务}_{训练/测试} = dict(...)
```

- 数据集缩写：见 [数据集名称对应表](#TODO)

- 算法任务：文本检测-det，文字识别-rec，关键信息提取-kie

- 训练/测试：数据集用于训练还是测试

以识别为例，使用 Syn90k 作为训练集，以 icdar2013 和 icdar2015 作为测试集配置如下：

```Python
# 识别数据集配置
mj_rec_train = dict(
    type='OCRDataset',
    data_root='data/rec/Syn90k/',
    data_prefix=dict(img_path='mnt/ramdisk/max/90kDICT32px'),
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

ic13_rec_test = dict(
    type='OCRDataset',
    data_root='data/rec/icdar_2013/',
    data_prefix=dict(img_path='Challenge2_Test_Task3_Images/'),
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)

ic15_rec_test = dict(
    type='OCRDataset',
    data_root='data/rec/icdar_2015/',
    data_prefix=dict(img_path='ch4_test_word_images_gt/'),
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)
```

<div id="pipeline_config"></div>

#### 数据流水线配置

MMOCR 中，数据集的构建与数据准备是相互解耦的。也就是说，`OCRDataset` 等数据集构建类负责完成标注文件的读取与解析功能；而数据变换方法（Data Transforms）则进一步实现了数据读取、数据增强、数据格式化等相关功能。

同时一般情况下训练和测试会存在不同的增强策略，因此一般会存在训练流水线（train_pipeline）和测试流水线（test_pipeline）。

训练流水线的数据增强流程通常为：数据读取(LoadImageFromFile)->标注信息读取(LoadXXXAnntation)->数据增强->数据格式化(PackXXXInputs)。

测试流水线的数据增强流程通常为：数据读取(LoadImageFromFile)->数据增强->标注信息读取(LoadXXXAnntation)->数据格式化(PackXXXInputs)。

更多信息可以参考[数据流水线](#TODO)

由于 OCR 任务的特殊性，一般情况下不同模型有不同数据增强的方式，相同模型在不同数据集一般也会有不同的数据增强方式。以 CRNN 为例：

```Python
# 数据增强
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='grayscale',
        file_client_args=dict(backend='disk'),
        ignore_empty=True,
        min_size=5),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(100, 32), keep_ratio=False),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='grayscale',
        file_client_args=dict(backend='disk')),
    dict(
        type='RescaleToHeight',
        height=32,
        min_width=32,
        max_width=None,
        width_divisor=16),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
```

<div id="dataloader_config"></div>

#### Dataloader 配置

主要为构造数据集加载器(dataloader)所需的配置信息，更多教程看参考[PyTorch 数据加载器](torch.data)。

```Python
# Dataloader 部分
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[mj_rec_train],
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[ic13_rec_test，ic15_rec_test],
        pipeline=test_pipeline))
test_dataloader = val_dataloader
```

### 模型相关配置

<div id="model_config"></div>

#### 网络配置

用于配置模型的网络结构，不同的算法任务有不同的网络结构，

##### 文本检测

文本检测主要包含几个部分:

- `data_preprocessor`: [数据处理器](mmocr.models.textdet.data_preprocessors.TextDetDataPreprocessor)
- `backbone`: 特征提取网络
- `neck`: 颈网络配置
- `det_head`: 检测头网络配置
  - `module_loss`: 模型损失函数配置
  - `postprocessor`: 模型预测结果后处理配置

我们以 DBNet 为例，介绍文字检测中模型配置：

```Python
model = dict(
    type='DBNet',
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32)
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=False,
        style='caffe'),
    neck=dict(
        type='FPNC', in_channels=[64, 128, 256, 512], lateral_channels=256),
    det_head=dict(
        type='DBHead',
        in_channels=256,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad')))
```

##### 文本识别

文本识别主要包含：

- `data_processor`: [数据预处理配置](mmocr.models.textrec.data_processors.TextRecDataPreprocessor)
- `preprocessor`: 网络预处理配置，如TPS等
- `backbone`:特征提取配置
- `encoder`: 编码器配置
- `decoder`: 解码器配置
  - `module_loss`: 解码器损失
  - `postprocessor`: 解码器后处理
  - `dictionary`: 字典配置

以 CRNN 为例：

```Python
# 模型部分
model = dict(
   type='CRNN',
   data_preprocessor=dict(
        type='TextRecogDataPreprocessor', mean=[127], std=[127])
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(
        type='CRNNDecoder',
        in_channels=512,
        rnn_flag=True,
        module_loss=dict(type='CTCModuleLoss', letter_case='lower'),
        postprocessor=dict(type='CTCPostProcessor'),
        dictionary=dict(
            type='Dictionary',
            dict_file='dicts/lower_english_digits.txt',
            with_padding=True)))
```

<div id="weight_config"></div>

#### 权重加载配置

可以通过 `load_from` 参数加载检查点（checkpoint）文件中的模型权重，只需要将 `load_from` 参数设置为检查点文件的路径即可。

用户也可通过设置 `resume=True` ，加载检查点中的训练状态信息来恢复训练。当 `load_from` 和 `resume=True` 同时被设置时，执行器将加载 `load_from` 路径对应的检查点文件中的训练状态。

如果仅设置 `resume=True`，执行器将会尝试从 `work_dir` 文件夹中寻找并读取最新的检查点文件

```Python
load_from = None # 加载checkpoint的路径
resume = False # 是否 resume
```

更多可以参考[加载权重或恢复训练](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/runner.md#%E5%8A%A0%E8%BD%BD%E6%9D%83%E9%87%8D%E6%88%96%E6%81%A2%E5%A4%8D%E8%AE%AD%E7%BB%83)与[OCR进阶技巧-断点恢复训练](https://mmocr.readthedocs.io/en/1.x/user_guides/train_test.html#resume-training-from-a-checkpoint)。

<div id="eval_config"></id>

### 评测配置

在模型验证和模型测试中，通常需要对模型精度做定量评测。MMOCR 通过评测指标(Metric)和评测器(Evaluator)来完成这一功能。更多可以参考[评测指标（Metric）和评测器（Evaluator）](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/metric_and_evaluator.md)

评测部分包含两个部分，评测器和评测指标。接下来我们分部分展开讲解。

#### 评测器

评测器主要用来管理多个数据集以及多个 `Metric`。针对单数据集与多数据集情况，评测器分为了单数据集评测器与多数据集评测器，这两种评测器均可管理多个 `Metric`.

单数据集评测器配置如下：

```Python
# 单个数据集 单个 Metric 情况
val_evaluator = dict(
    type='Evaluator',
    metrics=dict())

# 单个数据集 多个 Metric 情况
val_evaluator = dict(
    type='Evaluator',
    metrics=[...])
```

在实现中默认为单数据集评测器，因此对单数据集评测情况下，一般情况下只需配置评测器，即为

```Python
# 单个数据集 单个 Metric 情况
val_evaluator = dict()

# 单个数据集 多个 Metric 情况
val_evaluator = [...]
```

多数据集评测与单数据集评测存在两个位置上的不同：评测器类别与前缀。评测器类别必须为`MultiDatasetsEvaluator`且不能省略，前缀主要用来区分不同数据集在相同评测指标下的结果，请参考[多数据集评测](#TODO)。

假设我们需要在 IC13 和 IC15 情况下测试精度，则配置如下：

```Python
# 多个数据集，单个 Metric 情况
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=dict(),
    dataset_prefixes=['IC13', 'IC15'])

# 多个数据集，多个 Metric 情况
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[...],
    dataset_prefixes=['IC13', 'IC15'])
```

#### 评测指标

评测指标指不同度量精度的方法，同时可以多个评测指标共同使用，更多评测指标原理参考[评测指标](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/evaluation.md)，在 MMOCR 中不同算法任务有不同的评测指标。

文字检测: `HmeanIOU`

文字识别: `WordMetric`，`CharMetric`， `OneMinusNEDMetric`

关键信息提取: `F1Metric`

以文本检测为例说明，在单数据集评测情况下，使用单个 `Metric`：

```Python
val_evaluator = dict(type='HmeanIOUMetric')
```

以文本识别为例，多数据集使用多个 `Metric` 评测：

```Python
# 评测部分
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric')
    ],
    dataset_prefixes=['IC13', 'IC15'])
test_evaluator = val_evaluator
```

<div id="vis_config"></div>

### 可视化配置

每个任务配置该任务对应的可视化器。可视化器主要用于用户模型中间结果的可视化或存储，及 val 和 test 预测结果的可视化。同时可视化的结果可以通过可视化后端储存到不同的后端，比如 Wandb，TensorBoard 等。常用修改操作可见[可视化](visualization.md)。

文本检测的可视化默认配置如下：

```Python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextDetLocalVisualizer',  # 不同任务有不同的可视化器
    vis_backends=vis_backends,
    name='visualizer')
```

## 目录结构

`MMOCR` 所有配置文件都放置在 `configs` 文件夹下。为了避免配置文件过长，同时提高配置文件的可复用性以及清晰性，MMOCR 利用 Config 文件的继承特性，将配置内容的八个部分做了拆分。因为每部分均与算法任务相关，因此 MMOCR 对每个任务在 Config 中提供了一个任务文件夹，即 `textdet` (文字检测任务)、`textrec` (文字识别任务)、`kie` (关键信息提取)。同时各个任务算法配置文件夹下进一步划分为两个部分：`_base_` 文件夹与诸多算法文件夹：

1. `_base_` 文件夹下主要存放与具体算法无关的一些通用配置文件，各部分依目录分为常用的数据集、常用的训练策略以及通用的运行配置。

2. 算法配置文件夹中存放与算法强相关的配置项。算法配置文件夹主要分为两部分：

   1. 算法的模型与数据流水线：OCR 领域中一般情况下数据增强策略与算法强相关，因此模型与数据流水线通常置于统一位置。

   2. 算法在制定数据集上的特定配置：用于训练和测试的配置，将分散在不同位置的配置汇总。同时修改或配置一些在该数据集特有的配置比如batch size以及一些可能修改如数据流水线，训练策略等

最后的将配置内容中的各个模块分布在不同配置文件中，最终各配置文件内容如下:

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-lboi{border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>

<table class="tg">
<thead>
  <tr>
    <td class="tg-9wq8" rowspan="5">textdet<br></td>
    <td class="tg-lboi" rowspan="3">_base_</td>
    <td class="tg-9wq8">datasets</td>
    <td class="tg-0pky">icdar_datasets.py<br>ctw1500.py<br>...</td>
    <td class="tg-0pky"><a href="#dataset_config">数据集配置</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">schedulers</td>
    <td class="tg-0pky">schedule_adam_600e.py<br>...</td>
    <td class="tg-0pky"><a href="#schedule_config">训练策略配置</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">defaults_runtime.py<br></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky"><a href="#env_config">环境配置</a><br><a href="#hook_config">默认hook配置</a><br><a href="#log_config">日志配置</a> <br><a href="#weight_config">权重加载配置</a> <br><a href="#eval_config">评测配置</a> <br><a href="#vis_config">可视化配置</a></td>
  </tr>
  <tr>
    <td class="tg-lboi" rowspan="2">dbnet</td>
    <td class="tg-9wq8">_base_dbnet_resnet18_fpnc.py</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">-</span></td>
    <td class="tg-0pky"><a href="#model_config">网络配置</a> <br><a href="#pipeline_config">数据流水线</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">dbnet_resnet18_fpnc_1200e_icdar2015.py</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">-</span></td>
    <td class="tg-0pky"><a href="#dataloader_config">Dataloader 配置</a> <br><a href="#pipeline_config">数据流水线(Optional)</a></td>
  </tr>
</thead>
</table>

最终目录结构如下：

```Python
config
├── textdet
│   ├── _base_
│   │   ├── datasets
│   │   │   ├── icdar2015.py
│   │   │   ├── icdar2017.py
│   │   │   └── totaltext.py
│   │   ├── schedules
│   │   │   └── schedule_adam_600e.py
│   │   └── default_runtime.py
│   └── dbnet
│       ├── _base_dbnet_resnet18_fpnc.py
│       └── dbnet_resnet18_fpnc_1200e_icdar2015.py
├── textrecog
│   ├── _base_
│   │   ├── datasets
│   │   │   ├── icdar2015.py
│   │   │   ├── icdar2017.py
│   │   │   └── totaltext.py
│   │   ├── schedules
│   │   │   └── schedule_adam_base.py
│   │   └── default_runtime.py
│   └── crnn
│       ├── _base_crnn_mini-vgg.py
│       └── crnn_mini-vgg_5e_mj.py
└── Kie
    ├── _base_
    │   ├──datasets
    │   └── default_runtime.py
    └── sgdmr
        └── sdmgr_novisual_60e_wildreceipt_openset.py
```

## 配置文件以及权重命名规则

MMOCR 按照以下风格进行配置文件命名，代码库的贡献者需要遵循相同的命名规则。文件名总体分为四部分：算法信息，模块信息，训练信息和数据信息。逻辑上属于不同部分的单词之间用下划线 `'_'` 连接，同一部分有多个单词用短横线 `'-'` 连接。

```Python
{{算法信息}}_{{模块信息}}_{{训练信息}}_{{数据信息}}.py
```

- 算法信息(algorithm info)：算法名称，如 DBNet，CRNN 等

- 模块信息(module info)：按照数据流的顺序列举一些中间的模块，其内容依赖于算法任务，同时为了避免Config过长，会省略一些与模型强相关的模块。下面举例说明：

  - 对于文字检测任务和关键信息提取任务:

    ```Python
    {{算法信息}}_{{backbone}}_{{neck}}_{{head}}_{{训练信息}}_{{数据信息}}.py
    ```

    一般情况下 head 位置一般为算法专有的 head，因此一般省略。

  - 对于文本识别任务：

    ```Python
    {{算法信息}}_{{backbone}}_{{encoder}}_{{decoder}}_{{训练信息}}_{{数据信息}}.py
    ```

    一般情况下 encode 和 decoder 位置一般为算法专有，因此一般省略。

- 训练信息(training info)：训练策略的一些设置，包括 batch size，schedule 等

- 数据信息(data info)：数据集名称、模态、输入尺寸等，如 icdar2015，synthtext 等
