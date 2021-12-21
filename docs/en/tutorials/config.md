# Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

## Modify config through script arguments

When submitting jobs using "tools/train.py" or "tools/test.py", you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadImageFromNdarry'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.0.type=LoadImageFromNdarry`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to
  change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark \" is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Config Name Style

We follow the below style to name full config files (`configs/TASK/*.py`). Contributors are advised to follow the same style.

```
{model}_[ARCHITECTURE]_[schedule]_{dataset}.py
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type like `dbnet`, `crnn`, etc.
- `[ARCHITECTURE]`: expands some invoked modules following the order of data flow, and the content depends on the model framework. The following examples show how it is generally expanded.
  - For text detection tasks, key information tasks, and SegOCR in text recognition task: `{model}_[backbone]_[neck]_[schedule]_{dataset}.py`
  - For other text recognition tasks, `{model}_[backbone]_[encoder]_[decoder]_[schedule]_{dataset}.py`
  Note that `backbone`, `neck`, `encoder`, `decoder` are the names of modules, e.g. `r50`, `fpnocr`, etc.
- `{schedule}`: training schedule. For instance, `1200e` denotes 1200 epochs.
- `{dataset}`: dataset. It can either be the name of a dataset (`icdar2015`), or a collection of datasets for brevity (e.g. `academic` usually refers to a common practice in academia, which uses MJSynth + SynthText as training set, and IIIT5K, SVT, IC13, IC15, SVTP and CT80 as test set).

Most configs are composed of basic _primitive_ configs in `configs/_base_`, where each _primitive_ config in different subdirectory has a slightly different name style. We present them as follows.

- det_datasets, recog_datasets: `{dataset_name(s)}_[train|test].py`. If [train|test] is not specified, the config should contain both training and test set.

  There are two exceptions: toy_data.py and seg_toy_data.py. In recog_datasets, the first one works for most while the second one contains character level annotations and works for seg baseline only as of Dec 2021.
- det_models, recog_models: `{model}_[ARCHITECTURE].py`.
- det_pipelines, recog_pipelines: `{model}_pipeline.py`.
- schedules: `schedule_{optimizer}_{num_epochs}e.py`.

## Config Structure

For better config reusability, we break many of reusable sections of configs into `configs/_base_`. Now the directory tree of `configs/_base_` is organized as follows:

```
_base_
├── det_datasets
├── det_models
├── det_pipelines
├── recog_datasets
├── recog_models
├── recog_pipelines
└── schedules
```

These _primitive_ configs are categorized by their roles in a complete config. Most of model configs are making full use of _primitive_ configs by including them as parts of `_base_` section. For example, [dbnet_r18_fpnc_1200e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/5a8859fe6666c096b75fa44db4f6c53d81a2ed62/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py) takes five _primitive_ configs from `_base_`:

```python
_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_sgd_1200e.py',
    '../../_base_/det_models/dbnet_r18_fpnc.py',
    '../../_base_/det_datasets/icdar2015.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]
```

From these configs' names we can roughly know this config trains dbnet_r18_fpnc with sgd optimizer in 1200 epochs. It uses the origin dbnet pipeline and icdar2015 as the dataset. We encourage users to follow and take advantage of this convention to organize the config clearly and facilitate fair comparison across different _primitive_ configurations as well as models.

Please refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) for detailed documentation.


## Config File Structure

### model

The parameter `"model"` is a python dictionary in the configuration file, which mainly includes information such as network structure and loss function.

```{note}
The 'type' in the configuration file is not a constructed parameter, but a class name.
```

```{note}
We can also use models from MMDetection by adding `mmdet.` prefix to type name, or from other OpenMMLab projects in a similar way if their backbones are registered in registries.
```

- `type`: Model name.
- `backbone`: Backbone configs. [Common Backbones](https://mmocr.readthedocs.io/en/latest/api.html#common-backbones), [TextRecog Backbones](https://mmocr.readthedocs.io/en/latest/api.html#textrecog-backbones)
- `neck`: Neck network name. Refer to [TextDet Necks](https://mmocr.readthedocs.io/en/latest/api.html#textdet-necks), [TextRecog Necks](https://mmocr.readthedocs.io/en/latest/api.html#textrecog-necks).
- `bbox_head`: Head network name. Applicable to text detection, key information models and part of text recognition models. [TextDet Heads](https://mmocr.readthedocs.io/en/latest/api.html#textdet-dense-heads), [TextRecog Heads](https://mmocr.readthedocs.io/en/latest/api.html#textrecog-heads), [KIE Heads](https://mmocr.readthedocs.io/en/latest/api.html#module-mmocr.models.kie.heads).
  - `loss`: Loss function type. [TextDet Losses](https://mmocr.readthedocs.io/en/latest/api.html#textdet-losses), [KIE Losses](https://mmocr.readthedocs.io/en/latest/api.html#module-mmocr.models.kie.losses)
  - `postprocessor`: (TextDet only) Postprocess type. [TextDet Postprocessors](https://mmocr.readthedocs.io/en/latest/api.html#textdet-postprocess)
- `encoder`: Encoder configs. Applicable to text recognition models. [TextRecog Encoders](https://mmocr.readthedocs.io/en/latest/api.html#textrecog-encoders)
- `decoder`: Decoder configs. Applicable to text recognition models. [TextRecog Decoders](https://mmocr.readthedocs.io/en/latest/api.html#textrecog-decoders)
- `loss`: Loss configs. Applicable to some text recognition models.  [TextRecog Losses](https://mmocr.readthedocs.io/en/latest/api.html#textrecog-losses)
- `label_convertor`: Convert outputs between text, index and tensor. Applicable to text recognition models. [Label Convertors](https://mmocr.readthedocs.io/en/latest/api.html#textrecog-convertors)
- `max_seq_len`: The maximum sequence length of recognition results.Applicable to text recognition models.

### data & pipeline

The parameter `"data"` is a python dictionary in the configuration file, which mainly includes information to construct dataloader:

- `samples_per_gpu` : the BatchSize of each GPU when building the dataloader
- `workers_per_gpu` : the number of threads per GPU when building dataloader
- `train | val | test` : config to construct dataset
  - `type`: Dataset name. Check [dataset types](../dataset_types.md) for supported datasets.

The parameter `evaluation` is also a dictionary, which is the configuration information of `evaluation hook`, mainly including evaluation interval, evaluation index, etc.

```python
# dataset settings
dataset_type = 'IcdarDataset'  # dataset name，
data_root = 'data/icdar2015'  # dataset root
img_norm_cfg = dict(        # Image normalization config to normalize the input images
    mean=[123.675, 116.28, 103.53],  # Mean values used to pre-training the pre-trained backbone models
    std=[58.395, 57.12, 57.375],     # Standard variance used to pre-training the pre-trained backbone models
    to_rgb=True)                     # Whether to invert the color channel, rgb2bgr or bgr2rgb.
# train data pipeline
train_pipeline = [  # Training pipeline
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(
        type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True,  # Whether to use bounding box, True for detection
        with_mask=True,  # Whether to use instance mask, True for instance segmentation
        poly2mask=False),  # Whether to convert the polygon mask to instance mask, set False for acceleration and to save memory
    dict(
        type='Resize',  # Augmentation pipeline that resize the images and their annotations
        img_scale=(1333, 800),  # The largest scale of image
        keep_ratio=True
    ),  # whether to keep the ratio between height and width.
    dict(
        type='RandomFlip',  # Augmentation pipeline that flip the images and their annotations
        flip_ratio=0.5),  # The ratio or probability to flip
    dict(
        type='Normalize',  # Augmentation pipeline that normalize the input images
        mean=[123.675, 116.28, 103.53],  # These keys are the same of img_norm_cfg since the
        std=[58.395, 57.12, 57.375],  # keys of img_norm_cfg are used here as arguments
        to_rgb=True),
    dict(
        type='Pad',  # Padding config
        size_divisor=32),  # The number the padded images should be divisible
    dict(type='DefaultFormatBundle'),  # Default format bundle to gather data in the pipeline
    dict(
        type='Collect',  # Pipeline that decides which keys in the data should be passed to the detector
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(
        type='MultiScaleFlipAug',  # An encapsulation that encapsulates the testing augmentations
        img_scale=(1333, 800),  # Decides the largest scale for testing, used for the Resize pipeline
        flip=False,  # Whether to flip images during testing
        transforms=[
            dict(type='Resize',  # Use resize augmentation
                 keep_ratio=True),  # Whether to keep the ratio between height and width, the img_scale set here will be suppressed by the img_scale set above.
            dict(type='RandomFlip'),  # Thought RandomFlip is added in pipeline, it is not used because flip=False
            dict(
                type='Normalize',  # Normalization config, the values are from img_norm_cfg
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='Pad',  # Padding config to pad images divisible by 32.
                size_divisor=32),
            dict(
                type='ImageToTensor',  # convert image to tensor
                keys=['img']),
            dict(
                type='Collect',  # Collect pipeline that collect necessary keys for testing.
                keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=32,     # Batch size of a single GPU
    workers_per_gpu=2,      # Worker to pre-fetch data for each single GPU
    train=dict(            # train data config
        type=dataset_type,                  # dataset name
        ann_file=f'{data_root}/instances_training.json',  # Path to annotation file
        img_prefix=f'{data_root}/imgs',  # Path to images
        pipeline=train_pipeline),           # train data pipeline
    test=dict(             # test data config
        type=dataset_type,
        ann_file=f'{data_root}/instances_test.json',  # Path to annotation file
        img_prefix=f'{data_root}/imgs',  # Path to images
        pipeline=test_pipeline))
evaluation = dict(       # The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
    interval=1,          # Evaluation interval
    metric='hmean-iou')   # Metrics used during evaluation
```

### training schedule
Mainly include optimizer settings, `optimizer hook` settings, learning rate schedule and `runner` settings:
- `optimizer`: optimizer setting , support all optimizers in `pytorch`, refer to related [mmcv](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/optimizer/default_constructor.html#DefaultOptimizerConstructor) documentation.
- `optimizer_config`: `optimizer hook` configuration file, such as setting gradient limit, refer to related [mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8) code.
- `lr_config`: Learning rate scheduler, supports "CosineAnnealing", "Step", "Cyclic", etc. Refer to related [mmcv](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/lr_updater.html#LrUpdaterHook) documentation for more options.
- `runner`: For `runner`, please refer to `mmcv` for [`runner`](https://mmcv.readthedocs.io/en/latest/understand_mmcv/runner.html) introduction document.

```python
# he configuration file used to build the optimizer, support all optimizers in PyTorch.
optimizer = dict(type='SGD',         # Optimizer type
                lr=0.1,              # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
                momentum=0.9,        # Momentum
                weight_decay=0.0001) # Weight decay of SGD
# Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
optimizer_config = dict(grad_clip=None)  # Most of the methods do not use gradient clip
# Learning rate scheduler config used to register LrUpdater hook
lr_config = dict(policy='step',          # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
                 step=[30, 60, 90])      # Steps to decay the learning rate
runner = dict(type='EpochBasedRunner',   # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
            max_epochs=100)    # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
```

### runtime setting

This part mainly includes saving the checkpoint strategy, log configuration, training parameters, breakpoint weight path, working directory, etc..

```python
# Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
checkpoint_config = dict(interval=1)    # The save interval is 1
# config to register logger hook
log_config = dict(
    interval=100,                       # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook'),           # The Tensorboard logger is also supported
        # dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')   # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'             # The output level of the log.
resume_from = None             # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]      # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.
work_dir = 'work_dir'          # Directory to save the model checkpoints and logs for the current experiments.
```

## FAQ

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of fields in base configs.
You may refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#inherit-from-base-config-with-ignored-fields) for simple illustration.

### Use intermediate variables in configs

Some intermediate variables are used in the configs files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, user need to pass the intermediate variables into corresponding fields again.
For example, we usually want the data path to be a variable so that we

```python
dataset_type = 'IcdarDataset'
data_root = 'data/icdar2015'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)
```

### Use some fields in the base configs

Sometimes, you may refer to some fields in the `_base_` config, so as to avoid duplication of definitions. You can refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#reference-variables-from-base) for some more instructions.

This technique has been widely used in MMOCR's configs, where the main configs refer to the dataset and pipeline defined in _base_ configs by:

```python
train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}
```

Which assumes that its _base_ configs export datasets and pipelines in a way like:

```python
#  base dataset config
dataset_type = 'IcdarDataset'
data_root = 'data/icdar2015'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

train_list = [train]
test_list = [test]
```

```python
#  base pipeline config
train_pipeline = dict(...)
test_pipeline = dict(...)
```

## Deprecated train_cfg/test_cfg

The `train_cfg` and `test_cfg` are deprecated in config file, please specify them in the model config. The original config structure is as below.

```python
# deprecated
model = dict(
   type=...,
   ...
)
train_cfg=dict(...)
test_cfg=dict(...)
```

The migration example is as below.

```python
# recommended
model = dict(
   type=...,
   ...
   train_cfg=dict(...),
   test_cfg=dict(...),
)
```
