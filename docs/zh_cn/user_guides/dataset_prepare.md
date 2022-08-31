# 数据集准备

## 前言

经过数十年的发展，OCR 领域涌现出了一系列的相关数据集，这些数据集往往采用风格各异的格式来提供文本的标注文件，使得用户在使用这些数据集时不得不进行格式转换。MMOCR 支持了数十种常用的文本相关数据集，并提供了详细的数据下载及准备教程，同时我们给常用的[检测](./data_prepare/det.md)。[识别](./data_prepare/recog.md)及[关键信息抽取](./data_prepare/kie.md)提供了格式转换脚本，以方便用户将这些数据集转换为 MMOCR 支持的格式。

下面，我们对 MMOCR 内支持的各任务的数据格式进行简要的介绍。

- 如以下代码块所示，文本检测任务采用数据格式 `TextDetDataset`，其中存放了文本检测任务所需的边界盒标注、文件名等信息。我们在 `tests/data/det_toy_dataset/instances_test.json` 路径中提供了一个示例标注文件。

  ```json
    {
    "metainfo":
      {
        "dataset_type": "TextDetDataset",
        "task_name": "textdet",
        "category": [{"id": 0, "name": "text"}]
      },
    "data_list":
      [
        {
          "img_path": "test_img.jpg",
          "height": 640,
          "width": 640,
          "instances":
            [
              {
                "polygon": [0, 0, 0, 10, 10, 20, 20, 0],
                "bbox": [0, 0, 10, 20],
                "bbox_label": 0,
                "ignore": false
              }
            ],
            //...
        }
      ]
    }
  ```

- 如以下代码块所示，文本识别任务采用数据格式 `TextRecogDataset`，其中存放了文本识别任务所需的文本内容及图片路径等信息。我们在 `tests/data/rec_toy_dataset/labels.json` 路径中提供了一个示例标注文件。

  ```json
  {
    "metainfo":
      {
        "dataset_type": "TextRecogDataset",
        "task_name": "textrecog",
      },
    "data_list":
      [
        {
          "img_path": "test_img.jpg",
          "instances":
            [
              {
                "text": "GRAND"
              }
            ]
          }
      ]
  }
  ```

## 数据集下载及格式转换

以 ICDAR 2015 **文本检测数据集**的准备步骤为例，你可以依次执行以下步骤来完成数据集准备：

- 从 [ICDAR 官方网站](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载 ICDAR 2015 数据集。将训练集`ch4_training_word_images_gt.zip` 与测试集压缩包`ch4_test_word_images_gt.zip` 分别解压至路径 `data/icdar2015`。

  ```bash
  # 下载数据集
  wget https://rrc.cvc.uab.es/downloads/ch4_training_images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch4_test_images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge4_Test_Task1_GT.zip --no-check-certificate

  # 解压数据集
  mkdir imgs && mkdir annotations
  unzip ch4_training_images.zip -d imgs/training
  unzip ch4_training_localization_transcription_gt.zip -d annotations/training
  unzip ch4_test_images.zip -d imgs/test
  unzip Challenge4_Test_Task1_GT.zip -d annotations/test
  ```

- 使用 MMOCR 提供的格式转换脚本将原始的标注文件转换为 MMOCR 统一的数据格式

  ```bash
  python tools/dataset_converters/textdet/icdar_converter.py data/ic15/ -o data/icdar15/ --split-list training test -d icdar2015
  ```

- 完成上述步骤后，数据集标签将被转换为 MMOCR 使用的统一格式，文件目录结构如下：

  ```text
  data/ic15/
  ├── annotations
  │   ├── test
  │   └── training
  ├── imgs
  │   ├── test
  │   └── training
  ├── instances_test.json
  └── instances_training.json
  ```

## 数据集配置文件

### 单数据集训练

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

### 多数据集训练

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
```
