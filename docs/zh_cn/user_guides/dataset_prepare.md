# 数据集准备

## 数据集下载及格式转换

MMOCR 支持了数十种常用的文本[检测](../data_prepare/det.md)及[识别](../data_prepare/recog.md)数据集，并提供了详细的数据下载及准备教程。

以 ICDAR 2015 文本检测数据集的准备步骤为例，你可以依次执行以下步骤来完成数据集准备：

- 从 [ICDAR 官方网站](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载 ICDAR 2015 数据集。将训练集`ch4_training_word_images_gt.zip` 与测试集压缩包`ch4_test_word_images_gt.zip` 分别解压至路径 `data/icdar2015`。

- 下载 MMOCR 格式的标注文件 [train_label.json](#需要上传并更新链接) 和 [test_label.json](#需要上传并更新链接)。

- 完成上述步骤后，文件目录结构如下

```text
├── data/icdar2015
│   ├── train_label.json
│   ├── test_label.json
│   ├── ch4_training_word_images_gt
│   └── ch4_test_word_images_gt
```

## 数据集配置文件

### 单数据集训练及评测

在使用新的数据集时，我们需要对其图像、标注文件的路径等基础信息进行配置。`configs/xxx/_base_/datasets/` 路径下已预先配置了 MMOCR 中常用的数据集，这里我们以 ICDAR 2015 数据集为例（见 `configs/_base_/det_datasets/icdar2015.py`）：

```Python
ic15_det_data_root = 'data/det/icdar2015' # 数据集根目录

# 训练集配置
ic15_det_train = dict(
    type='OCRDataset',
    data_root=ic15_det_data_root,                        # 数据根目录
    ann_file='instances_training.json',                  # 标注文件名称
    data_prefix=dict(img_path='imgs/'),                  # 图片路径前缀
    filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 数据过滤
    pipeline=None)
# 测试集配置
ic15_det_test = dict(
    type='OCRDataset',
    data_root=ic15_det_data_root,
    ann_file='instances_test.json',
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=None)
```

在配置好数据集后，我们还需要在相应的算法模型配置文件中导入想要使用的数据集。例如，在 ICDAR 2015 数据集上训练 "DBNet_R18" 模型：

```Python
_base_ = [
    '_base_dbnet_r18_fpnc.py',
    '../_base_/datasets/icdar2015.py',  # 导入数据集配置文件
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

ic15_det_train = _base_.ic15_det_train            # 指定训练集
ic15_det_train.pipeline = _base_.train_pipeline   # 指定训练集使用的数据流水线
ic15_det_test = _base_.ic15_det_test              # 指定测试集
ic15_det_test.pipeline = _base_.test_pipeline     # 指定测试集使用的数据流水线

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_det_train)    # 在 train_dataloader 中指定使用的训练数据集

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ic15_det_test)    # 在 val_dataloader 中指定使用的验证数据集

test_dataloader = val_dataloader
```

### 多数据集训练及评测

此外，基于 [`ConcatDataset`](mmocr.datasets.ConcatDataset)，用户还可以使用多个数据集组合来训练或测试模型。用户只需在配置文件中将 dataloader 中的 dataset 类型设置为 `ConcatDataset`，并指定对应的数据集列表即可。

```Python
train_list = [ic11, ic13, ic15]
train_dataloader = dict(
    dataset=dict(
        type='ConcatDataset', datasets=train_list, pipeline=train_pipeline))
```

例如，以下配置使用了 MJSynth 数据集进行训练，并使用 6 个学术数据集（CUTE80, IIIT5K, SVT, SVTP, ICDAR2013, ICDAR2015）进行测试。

```Python
_base_ = [ # 导入所有需要使用的数据集配置
    '../_base_/datasets/mjsynth.py',
    '../_base_/datasets/cute80.py',
    '../_base_/datasets/iiit5k.py',
    '../_base_/datasets/svt.py',
    '../_base_/datasets/svtp.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_crnn_mini-vgg.py',
]

# 训练集列表
train_list = [_base_.mj_rec_train]
# 测试集列表
test_list = [
    _base_.cute80_rec_test, _base_.iiit5k_rec_test, _base_.svt_rec_test,
    _base_.svtp_rec_test, _base_.ic13_rec_test, _base_.ic15_rec_test
]

# 使用 ConcatDataset 来级联列表中的多个数据集
train_dataset = dict(
       type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
       type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=192 * 4,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader

# 此处的 prefix 顺序需要与 test_list 列表中的数据集顺序一致
val_evaluator = dict(
    dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
test_evaluator = val_evaluator
```

需要注意的是，在使用多数据集测试时，我们需要将 evaluator 的类型指定为 [`MultiDatasetsEvaluator`](mmocr.evaluation.MultiDatasetsEvaluator)。并且，`dataset_prefixes` 中的名称顺序需要与 `test_list` 列表中的数据集顺序保持一致，这样在输出测试精度时，每项得分的前缀名才能被正确地打印在日志中。
