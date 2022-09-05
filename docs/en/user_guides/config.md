# Config

MMOCR mainly uses Python files as configuration files. The design of its configuration file system integrates the ideas of modularity and inheritance to facilitate various experiments.

## Common Usage

```{note}
This section is recommended to be read together with the primary usage in {external+mmengine:doc}`Config <tutorials/config>`.
```

There are three most common operations in MMOCR: inheritance of configuration files, reference to `_base_` variables, and modification of `_base_` variables. Config provides two syntaxes for inheriting and modifying `_base_`, one for Python, Json, and Yaml, and one for Python configuration files only. In MMOCR, we **prefer the Python-only syntax**, so this will be the basis for further description.

The `configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py` is used as an example to illustrate the three common uses.

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

### Configuration Inheritance

There is an inheritance mechanism for configuration files, i.e. one configuration file A can use another configuration file B as its base and inherit all the fields directly from it, thus avoiding a lot of copy-pasting.

In `dbnet_resnet18_fpnc_1200e_icdar2015.py` you can see that

```Python
_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]
```

The above statement reads all the base configuration files in the list, and all the fields in them are loaded into `dbnet_resnet18_fpnc_1200e_icdar2015.py`. We can see the structure of the configuration file after it has been parsed by running the following statement in a Python interpretation.

```Python
from mmengine import Config
db_config = Config.fromfile('configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py')
print(db_config)
```

It can be found that the parsed configuration contains all the fields and information in the base configuration.

```{note}
Please note: Variables with the same name cannot exist in each _base_ profile.
```

### `_base_` Variable References

Sometimes we may need to reference some fields in the `_base_` configuration directly in order to avoid duplicate definitions. Suppose we want to get the variable `pseudo` in the `_base_` configuration, we can get the variable in the `_base_` configuration directly via `_base_.pseudo`.

This syntax has been used extensively in the configuration of MMOCR, and the dataset and pipeline configurations for each model in MMOCR are referenced in the *_base_* configuration. For example,

```Python
ic15_det_train = _base_.ic15_det_train
# ...
train_dataloader = dict(
    # ...
    dataset=ic15_det_train)
```

### `_base_` Variable Modification

In MMOCR, different algorithms usually have different pipelines in different datasets, so there are often scenarios to modify the `pipeline` in the dataset. There are also many scenarios where you need to modify variables in the `_base_` configuration, for example, modifying the training strategy of an algorithm, replacing some modules of an algorithm(backbone, etc.). Users can directly modify the referenced `_base_` variables using Python syntax. For dict, we also provide a method similar to class attribute modification to modify the contents of the dictionary directly.

1. Dictionary

   Here is an example of modifying a `pipeline` in a dataset.

   The dictionary can be modified using Python syntax:

   ```Python
   # Get the dataset in _base_
   ic15_det_train = _base_.ic15_det_train
   # You can modify the variables directly with Python's update
   ic15_det_train.update(pipeline=_base_.train_pipeline)
   ```

   Changes can also be made using the methods of the class attribute.

   ```Python
   # Get the dataset in _base_
   ic15_det_train = _base_.ic15_det_train
   # The class property method is modified
   ic15_det_train.pipeline = _base_.train_pipeline
   ```

2. List

   Suppose the variable `pseudo = [1, 2, 3]` in the `_base_` configuration needs to be modified to `[1, 2, 4]`:

   ```Python
   # pseudo.py
   pseudo = [1, 2, 3]
   ```

   Can be rewritten directly as.

   ```Python
   _base_ = ['pseudo.py']
   pseudo = [1, 2, 4]
   ```

   Or modify the list using Python syntax:

   ```Python
   _base_ = ['pseudo.py']
   pseudo = _base_.pseudo
   pseudo[2] = 4
   ```

### Command Line Modification

Sometimes we only want to fix part of the configuration and do not want to modify the configuration file itself. For example, if you want to change the learning rate during an experiment but do not want to write a new configuration file, you can pass in parameters on the command line to override the relevant configuration.

We can pass `--cfg-options` on the command line and modify the corresponding fields directly with the arguments after it. For example, if we want to modify the learning rate while running train, we just need to execute on the command line.

```Shell
python tools/train.py example.py --cfg-options optim_wrapper.optimizer.lr=1
```

For more detailed usage, refer to {external+mmengine:doc}`Command Line Modification <tutorials/config>`.

## Configuration Content

Through the configuration file with the registrar, MMOCR can modify the training parameters as well as the model configuration without invading the code. Specifically, users can customize the following modules in the configuration file: environment configuration, Hook configuration, logging configuration, training strategy configuration, data-related configuration, model-related configuration, evaluation configuration, and visualization configuration.

This document will take the text detection algorithm `DBNet` and the text recognition algorithm `CRNN` as examples to introduce the contents of Config in detail.

<div id="env_config"></div>

### Environment Configuration

```Python
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
random_cfg = dict(seed=None)
```

There are three main components:

- Set the default `scope` of all registrars to `mmocr`, ensuring that all modules are searched first from the `MMOCR` codebase. If the module does not exist, the search will continue from the upstream algorithm libraries `MMEngine` and `MMCV`, see {external+mmengine:doc}`Registry <tutorials/registry>` for more details.

- `env_cfg` sets the distributed environment configuration, see {external+mmengine:doc}`Runner <tutorials/runner>` for more details.

- `random_cfg` set random seeds for numpy, torch, cudnn, etc., see {external+mmengine:doc}`Runner <tutorials/runner>` for more details.

<div id="hook_config"></div>

### Hook Configuration

Hooks are divided into two main parts, default hooks, which are required for all tasks to run, and custom hooks, which generally serve specific algorithms or specific tasks (there are no custom hooks in MMOCR so far).

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

Here is a brief description of a few hooks that may change frequently. For a general modification method, refer to [Modify configuration](#base-variable-modification).

- `LoggerHook`: Used to configure the behavior of the logger. For example, by modifying `interval` you can control the interval of log printing, so that the log is printed once per `interval` iteration, for more settings refer to [LoggerHook API](mmengine.hooks.LoggerHook).

- `CheckpointHook`: Used to configure model breakpoint saving related behavior, such as saving optimal weights, saving latest weights, etc. You can also modify `interval` to control the checkpoint saving interval. More settings can be found in [CheckpointHook API](mmengine.hooks.CheckpointHook)

- `VisualizationHook`: Used to configure visualization-related behavior, such as visualizing predicted results during validation or testing, default is off. This Hook also depends on [Visualizaiton Configuration](#visualizaiton-configuration). You can refer to [Visualizer](visualization.md) for more details. For more configuration, you can refer to [VisualizationHook API](mmocr.engine.hooks.VisualizationHook).

If you want to learn more about the configuration of the default hooks and their functions, you can refer to {external+mmengine:doc}`Hooks <tutorials/hook>`.

<div id="log_config"></div>

### Log Configuration

This section is mainly used to configure the logging configuration level and the logging processor.

```Python
log_level = 'INFO' # 日志记录等级
log_processor = dict(type='LogProcessor',
                        window_size=10,
                        by_epoch=True)
```

- The configuration level of logging is the same as that of {external+python:doc}`logging <library/logging>`.

- The log processor is mainly used to control the format of the output, detailed functions can be found in {external+mmengine:doc}`logging <advanced_tutorials/logging>`.

  - `by_epoch=True` indicates that the logs are output according to epoch, and the log format needs to be consistent with the `type='EpochBasedTrainLoop'` parameter in `train_cfg`. For example, if you want to output logs by iteration number, you need to make ` by_epoch=False` in `log_processor` and `type='IterBasedTrainLoop'` in `train_cfg`.

  - `window_size` indicates the smoothing window of the loss, i.e. the average value of the various losses for the last `window_size` iterations. the final loss value printed in logger is the average value after the various losses.

  <div id="schedule_config"></div>

### Training Strategy Configuration

This section mainly contains optimizer settings, learning rate strategy and `Loop` settings.

For different algorithm tasks (text detection, text recognition, key information extraction), there are usually common tuning strategies for their own tasks. The corresponding configurations involved for `CRNN` in text recognition are listed here.

```Python
# optimizer
optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adadelta', lr=1.0))
param_scheduler = [dict(type='ConstantLR', factor=1.0)]
train_cfg = dict(type='EpochBasedTrainLoop',
                    max_epochs=5, # train epochs
                    val_interval=1) # val interval
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

- `optim_wrapper` : It contains two main parts, OptimWrapper and Optimizer. Detailed usage information can be found in {external+mmengine:doc}`MMEngine Optimizer Wrapper <tutorials/optim_wrapper>`.

  - The Optimizer wrapper supports different training strategies, including mixed-accuracy training (AMP), gradient accumulation, and gradient truncation.

  - All PyTorch optimizers are supported in the optimizer settings. All supported optimizers are available in {external+torch:ref}`PyTorch Optimizer List <optim:algorithms>`.

- `param_scheduler` : learning rate tuning strategy, supports most of the learning rate schedulers in PyTorch, such as `ExponentialLR`, `LinearLR`, `StepLR`, `MultiStepLR`, etc., and is used in much the same way, see [scheduler interface](mmengine.optim.scheduler), and more features can be found in the {external+mmengine:doc}`Optimizer Parameter Tuning Strategy <tutorials/param_scheduler>`.

- `train/test/val_cfg` : the execution flow of the task, MMEngine provides four kinds of flow: `EpochBasedTrainLoop`, `IterBasedTrainLoop`, `ValLoop`, `TestLoop` More can be found in {external+mmengine:doc}`loop controller <advanced_tutorials/runner>`.

### Data-related Configuration

<div id="dataset_config"></id>

#### Dataset Configuration

It is mainly used to configure two directions.

- The image of the dataset and the location of the annotation file.

- Data augmentation related configurations. In the OCR domain, data augmentation is usually strongly associated with the model.

More parameter configurations can be found in [Data Base Class](#TODO).

The naming convention for dataset fields in MMOCR is

```Python
{dataset name abbreviation}_{algorithm task}_{training/testing} = dict(...)
```

- Dataset abbreviations: see [table corresponding to dataset names](#TODO)

- Algorithm tasks: text detection-det, text recognition-rec, key information extraction-kie

- Training/testing: dataset for training or testing

For the recognition example, Syn90k is used as the training set and icdar2013 and icdar2015 as the test sets are configured as follows.

```Python
# Recognition of dataset configuration
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

#### Data Pipeline Configuration

In MMOCR, dataset construction and data preparation are decoupled from each other. In other words, dataset construction classes such as `OCRDataset` are responsible for reading and parsing annotation files, while Data Transforms further implement data reading, data enhancement, data formatting and other related functions.

In general, there are different enhancement strategies for training and testing, so there are generally training_pipeline and testing_pipeline.

The data enhancement process of the training pipeline is usually: data reading (LoadImageFromFile) -> annotation information reading (LoadXXXAnntation) -> data enhancement -> data formatting (PackXXXInputs).

The data enhancement flow of the test pipeline is usually: Data Read (LoadImageFromFile) -> Data Enhancement -> Annotation Read (LoadXXXAnntation) -> Data Formatting (PackXXXInputs).

More information can be found in [Data Pipeline](../basic_concepts/transforms.md)

Due to the specificity of the OCR task, in general different models have different ways of data augmentation, and the same model will generally have different ways of data augmentation in different datasets. Take `CRNN` as an example.

```Python
# Data Augmentation
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

#### Dataloader Configuration

The main configuration information needed to construct the dataset loader (dataloader), see {external+torch:doc}`PyTorch DataLoader <data>` for more tutorials.

```Python
# Dataloader
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

### Model-related Configuration

<div id="model_config"></div>

#### Network Configuration

The network structure used to configure the model. Different network structures for different algorithmic tasks.

##### Text Detection

Text detection consists of several parts:

- `data_preprocessor`: [data_preprocessor](mmocr.models.textdet.data_preprocessors.TextDetDataPreprocessor)
- `backbone`: feature extraction network
- `neck`: neck network configuration
- `det_head`: detection head network configuration
  - `module_loss`: model loss function configuration
  - `postprocessor`: model prediction result post-processing configuration

We present the model configuration in text detection using DBNet as an example.

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

##### Text Recognition

Text recognition mainly contains.

- `data_processor`: [data preprocessor configuration](mmocr.models.textrec.data_processors.TextRecDataPreprocessor)
- `preprocessor`: network preprocessor configuration, e.g. TPS
- `backbone`: feature extraction configuration
- `encoder`: encoder configuration
- `decoder`: decoder configuration
  - `module_loss`: decoder loss
  - `postprocessor`: decoder postprocessor
  - `dictionary`: dictionary configuration

Using CRNN as an example.

```Python
# model
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

#### Weight Loading Configuration

The model weights in the checkpoint file can be loaded via the `load_from` parameter, simply by setting the `load_from` parameter to the path of the checkpoint file.

You can also resume training by setting `resume=True` to load the training status information in the checkpoint. When both `load_from` and `resume=True` are set, the actuator will load the training state from the checkpoint file corresponding to the `load_from` path.

If only `resume=True` is set, the executor will try to find and read the latest checkpoint file from the `work_dir` folder

```Python
load_from = None # Path to load checkpoint
resume = False # whether resume
```

More can be found in {external+mmengine:doc}`Load Weights or Recover Training <tutorials/runner>` and [OCR Advanced Tips - Breakpoint Recovery Training](train_test.md#resume-training-from-a-checkpoint).

<div id="eval_config"></id>

### Evaluation Configuration

In model validation and model testing, quantitative measurement of model accuracy is often required. MMOCR performs this function by means of `Metric` and `Evaluator`. For more information, please refer to {external+mmengine:doc}`Metric and Evaluator <tutorials/evaluation>`.

#### Evaluator

Raters are mainly used to manage multiple datasets and multiple `Metric`s. For single and multiple dataset cases, there are single and multiple dataset reviewers, both of which can manage multiple `Metric`.

The single-dataset evaluator is configured as follows.

```Python
# Single Dataset Single Metric
val_evaluator = dict(
    type='Evaluator',
    metrics=dict())

# Single Dataset Multiple Metric
val_evaluator = dict(
    type='Evaluator',
    metrics=[...])
```

`MultiDatasetsEvaluator` differs from single-dataset evaluation in two positions: rater category and prefix. The evaluator category must be `MultiDatasetsEvaluator` and cannot be omitted. The prefix is mainly used to distinguish the results of different datasets with the same evaluation metrics, see [MultiDatasetsEvaluation](../basic_concepts/evaluation.md).

Assuming that we need to test accuracy in the IC13 and IC15 cases, the configuration is as follows.

```Python
#  Multiple datasets, single Metric
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=dict(),
    dataset_prefixes=['IC13', 'IC15'])

# Multiple datasets, multiple Metric
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[...],
    dataset_prefixes=['IC13', 'IC15'])
```

#### Metric

Metrics refer to different measures of accuracy, while multiple metrics can be used together, for more metrics principles refer to {external+mmengine:doc}`Metric <tutorials/evaluation>`, there are different metrics for different algorithmic tasks in MMOCR.

Text detection: [`HmeanIOUMetric`](mmocr.evaluation.metrics.HmeanIOUMetric)

Text recognition: [`WordMetric`](mmocr.evaluation.metrics.WordMetric), [`CharMetric`](mmocr.evaluation.metrics.CharMetric), [`OneMinusNEDMetric`](mmocr.evaluation.metrics.OneMinusNEDMetric)

Key information extraction: `F1Metric`

Text detection as an example, using a single `Metric` in the case of single dataset evaluation.

```Python
val_evaluator = dict(type='HmeanIOUMetric')
```

Taking text recognition as an example, multiple datasets are evaluated using multiple `Metric`.

```Python
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

### Visualizaiton Configuration

Each task is configured with a visualizer corresponding to that task. The visualizer is mainly used for visualizing or storing intermediate results of user models and visualizing val and test prediction results. The visualization results can also be stored in different backends such as WandB, TensorBoard, etc. through the visualization backend. Commonly used modification operations can be found in [visualization](visualization.md).

The default configuration of visualization for text detection is as follows.

```Python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextDetLocalVisualizer',  # Different visualizers for different tasks
    vis_backends=vis_backends,
    name='visualizer')
```

## Directory Structure

All configuration files of `MMOCR` are placed under the `configs` folder. To avoid long configuration files and to improve the reusability and clarity of configuration files, MMOCR takes advantage of the inheritance property of Config files to split the eight sections of configuration content. Since each section is related to an algorithm task, MMOCR provides a task folder for each task in Config, namely `textdet` (text detection task), `textrec` (text recognition task), and `kie` (key information extraction). Also the individual task algorithm configuration folder is further divided into two parts: `_base_` folder and many algorithm folders.

1. the `_base_` folder mainly stores some general configuration files unrelated to specific algorithms, and each section is divided into common datasets, common training strategies and common running configurations by directory.

2. The algorithm configuration folder stores configuration items that are strongly related to the algorithm. The algorithm configuration folder is divided into two main sections.

   1. the model and data pipeline of the algorithm: in the OCR domain, in general, data enhancement strategies are strongly related to the algorithm, so the model and data pipeline are usually placed in a unified location.

   2. Algorithm-specific configurations on the developed dataset: configurations for training and testing, aggregating configurations that are scattered in different locations. Also modify or configure some configurations specific to the dataset such as batch size and some possible modifications such as data pipeline, training strategy, etc.

The final configuration content of each module is distributed in different configuration files, and the final content of each configuration file is as follows:

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
    <td class="tg-0pky"><a href="#dataset_config">Dataset configuration</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">schedulers</td>
    <td class="tg-0pky">schedule_adam_600e.py<br>...</td>
    <td class="tg-0pky"><a href="#schedule_config">Training Strategy Configuration</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">defaults_runtime.py<br></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky"><a href="#env_config">Environment Configuration</a><br><a href="#hook_config">Hook Configuration</a><br><a href="#log_config">Log Configuration</a> <br><a href="#weight_config">Weight Loading Configuration</a> <br><a href="#eval_config">Evaluation Configuration</a> <br><a href="#vis_config">Visualizaiton Configuration</a></td>
  </tr>
  <tr>
    <td class="tg-lboi" rowspan="2">dbnet</td>
    <td class="tg-9wq8">_base_dbnet_resnet18_fpnc.py</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">-</span></td>
    <td class="tg-0pky"><a href="#model_config">Network Configuration</a> <br><a href="#pipeline_config">Data Pipeline Configuration</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">dbnet_resnet18_fpnc_1200e_icdar2015.py</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">-</span></td>
    <td class="tg-0pky"><a href="#dataloader_config">Dataloader Configuration</a> <br><a href="#pipeline_config">Data Pipeline Configuration(Optional)</a></td>
  </tr>
</thead>
</table>

The final directory structure is as follows.

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
└── kie
    ├── _base_
    │   ├──datasets
    │   └── default_runtime.py
    └── sgdmr
        └── sdmgr_novisual_60e_wildreceipt_openset.py
```

## Naming Rules

MMOCR follows the following style for profile naming, and contributors to the code base need to follow the same naming rules. The file names are generally divided into four sections: algorithm information, module information, training information, and data information. Words that logically belong to different sections are connected by an underscore `'_'`, and multiple words in the same section are connected by a short horizontal line `'-'`.

```Python
{{algorithm info}}_{{module info}}_{{training info}}_{{data info}}.py
```

- algorithm info: the name of the algorithm, such as DBNet, CRNN, etc.

- Module info: list some intermediate modules in the order of data flow, whose content depends on the algorithm task, and some modules strongly related to the model will be omitted to avoid overly long Config. The following examples are given.

  - For the text detection task and the key information extraction task :

    ```Python
    {{algorithm info}}_{{backbone}}_{{neck}}_{{head}}_{{training info}}_{{data info}}.py
    ```

    In general the head position is usually the algorithm's proprietary head, so it is usually omitted.

  - For text recognition tasks.

    ```Python
    {{algorithm info}}_{{backbone}}_{{encoder}}_{{decoder}}_{{training info}}_{{data info}}.py
    ```

    In general the encode and decoder positions are generally exclusive to the algorithm, so they are generally omitted.

- training info: some settings of the training strategy, including batch size, schedule, etc.

- data info: dataset name, modality, input size, etc., such as icdar2015, synthtext, etc.
