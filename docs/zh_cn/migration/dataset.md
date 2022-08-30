# 数据集迁移

在 OpenMMLab 2.0 系列算法库基于 [MMEngine](https://github.com/open-mmlab/mmengine) 设计了统一的[数据集基类](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/basedataset.md)，并制定了数据集标注文件规范。基于此，我们在 MMOCR 1.0 版本中重构了 OCR 任务数据集基类 [`OCRDataset`](mmocr.datasets.OCRDataset)。以下文档将介绍 MMOCR 中新旧数据集格式的区别，以及如何将旧数据集迁移至新版本中。对于暂不方便进行数据迁移的用户，我们也在[第三节](#兼容性)提供了临时的代码兼容方案。

```{note}
关键信息抽取任务仍采用原有的 WildReceipt 数据集标注格式。
```

## 旧版数据格式回顾

针对不同任务，MMOCR 0.x 版本实现了多种不同的数据集类型，如文本检测任务的 `IcdarDataset`，`TextDetDataset`；文本识别任务的 `OCRDataset`，`OCRSegDataset` 等。而不同的数据集类型同时还可能存在多种不同的标注及文件存储后端，如 `.txt`、`.json`、`.jsonl` 等，使得用户在自定义数据集时需要配置各类数据加载器 (`Loader`) 以及数据解析器 (`Parser`)。这不仅增加了用户的使用难度，也带来了许多问题和隐患。例如，以 `.txt` 格式存储的简单 `OCDDataset` 在遇到包含空格的文本标注时将会报错。

### 文本检测

文本检测任务中，`IcdarDataset` 采用了与通用目标检测 COCO 数据集一致的标注格式。

```json
{
  "images": [
    {
      "id": 1,
      "width": 800,
      "height": 600,
      "file_name": "test.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [0,0,10,10],
      "segmentation": [
          [0,0,10,0,10,10,0,10]
      ],
      "area": 100,
      "iscrowd": 0
    }
  ]
}
```

而 `TextDetDataset` 则采用了 JSON Line 的存储格式，将类似 COCO 格式的标签转换成文本存放在 `.txt` 或 `.jsonl` 格式文件中。

```text
{"file_name": "test/img_2.jpg", "height": 720, "width": 1280,  "annotations": [{"iscrowd": 0, "category_id": 1, "bbox": [602.0, 173.0,  33.0, 24.0], "segmentation": [[602, 173, 635, 175, 634, 197, 602,  196]]}, {"iscrowd": 0, "category_id": 1, "bbox": [734.0, 310.0, 58.0,  54.0], "segmentation": [[734, 310, 792, 320, 792, 364, 738, 361]]}]}
{"file_name": "test/img_5.jpg", "height": 720, "width": 1280,  "annotations": [{"iscrowd": 1, "category_id": 1, "bbox": [405.0, 409.0,  32.0, 52.0], "segmentation": [[408, 409, 437, 436, 434, 461, 405,  433]]}, {"iscrowd": 1, "category_id": 1, "bbox": [435.0, 434.0, 8.0,  33.0], "segmentation": [[437, 434, 443, 440, 441, 467, 435, 462]]}]}
```

### 文本识别

对于文本识别任务，MMOCR 0.x 版本中存在两种数据标注格式。其中 `.txt` 格式的标注文件每一行共有两个字段，分别存放了图片名以及标注的文本内容，并以空格分隔。

```text
img1.jpg OpenMMLab
img2.jpg MMOCR
```

而 JSON Line 格式则使用 `json.dumps` 将 JSON 格式的标注转换为文本内容后存放在 .jsonl 文件中，其内容形似一个字典，将文件名和文本标注信息分别存放在 `filename` 和 `text` 字段中。

```json
{"filename": "img1.jpg", "text": "OpenMMLab"}
{"filename": "img2.jpg", "text": "MMOCR"}
```

## 新版数据格式

为解决 0.x 版本中数据集格式过于混杂的情况，MMOCR 1.x 采用了基于 MMEngine 设计的统一数据标准。每一个数据标注文件存放在 `.json` 文件中，并使用类似字典的格式分别存放了数据集的元信息（`metainfo`）与具体的标注内容（`data_list`）。

```json
{
  "metainfo":
    {
      "classes": ("cat", "dog"),
      ...
    },
  "data_list":
    [
      {
        "img_path": "xxx/xxx_0.jpg",
        "img_label": 0,
        ...
      },
      ...
    ]
}
```

基于此，我们针对 MMOCR 特有的任务设计了 `TextDetDataset`、`TextRecogDataset`。

### 文本检测

#### 新版格式介绍

`TextDetDataset` 中存放了文本检测任务所需的边界盒标注、文件名等信息。由于文本检测任务中只有 1 个类别，因此我们将其类别 id 默认设置为 0，而背景类则为 1。`tests/data/det_toy_dataset/instances_test.json` 中存放了一个文本检测任务的数据标注示例，用户可以参考该文件来将自己的数据集转换为我们支持的格式。

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
              "ignore": False
            }，
            ...
          ]
      }
    ]
}
```

#### 迁移脚本

为帮助用户将旧版本标注文件迁移至新格式，我们提供了迁移脚本。使用方法如下：

```bash
python tools/dataset_converters/textdet/data_migrator.py ${IN_PATH} ${OUT_PATH}
```

| 参数     | 类型                             | 说明                                                                                                                                               |
| -------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| in_path  | str                              | （必须）旧版标注的路径                                                                                                                             |
| out_path | str                              | （必须）新版标注的路径                                                                                                                             |
| --task   | 'auto', 'textdet', 'textspotter' | 指定输出数据集标注的所兼容的任务。若指定为 textdet ，则不会转存 coco 格式中的 text 字段。默认为 auto，即根据旧版标注的格式自动决定输出的标注格式。 |

### 文本识别

#### 新版格式介绍

`TextRecogDataset` 中存放了文本识别任务所需的文本内容，通常而言，文本识别数据集中的每一张图片都仅包含一个文本实例。我们在 `tests/data/rec_toy_dataset/labels.json` 提供了一个简单的识别数据格式示例，用户可以参考该文件以进一步了解其中的细节。

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

#### 迁移脚本

为帮助用户将旧版本标注文件迁移至新格式，我们提供了迁移脚本。使用方法如下：

```bash
python tools/dataset_converters/textrecog/data_migrator.py ${IN_PATH} ${OUT_PATH} --format ${txt, jsonl, lmdb}
```

| 参数     | 类型                   | 说明                       |
| -------- | ---------------------- | -------------------------- |
| in_path  | str                    | （必须）旧版标注的路径     |
| out_path | str                    | （必须）新版标注的路径     |
| --format | 'txt', 'jsonl', 'lmdb' | 指定旧版数据集标注的格式。 |

## 兼容性

考虑到用户对数据迁移所需的成本，我们在 MMOCR 1.x 版本中暂时对 MMOCR 0.x  旧版本格式进行了兼容。

```{note}
用于兼容旧数据格式的代码和组件可能在未来的版本中被完全移除。因此，我们强烈建议用户将数据集迁移至新的数据格式标准。
```

具体而言，我们提供了三个临时的数据集类 [IcdarDataset](mmocr.datasets.IcdarDataset), [RecogTextDataset](mmocr.datasets.RecogTextDataset), [RecogLMDBDataset](mmocr.datasets.RecogLMDBDataset) 来兼容旧格式的标注文件。分别对应了 MMOCR 0.x 版本中的文本检测数据集 `IcdarDataset`，`.txt`、`.jsonl` 和 `LMDB` 格式的文本识别数据标注。其使用方式与 0.x 版本一致。

1. [IcdarDataset](mmocr.datasets.IcdarDataset) 支持 0.x 版本文本检测任务的 COCO 标注格式。只需要在 `configs/textdet/_base_/datasets` 中添加新的数据集配置文件，并指定其数据集类型为 `IcdarDataset` 即可。

   ```python
     data_root = 'data/det/icdar2015'

     train_dataset = dict(
         type='IcdarDataset',
         data_root=data_root,
         ann_file='instances_training.json',
         data_prefix=dict(img_path='imgs/'),
         filter_cfg=dict(filter_empty_gt=True, min_size=32),
         pipeline=None)
   ```

2. [RecogTextDataset](mmocr.datasets.RecogTextDataset) 支持 0.x 版本文本识别任务的 `txt` 和 `jsonl` 标注格式。只需要在 `configs/textrecog/_base_/datasets` 中添加新的数据集配置文件，并指定其数据集类型为 `RecogTextDataset` 即可。例如，以下示例展示了如何配置并读取 toy dataset 中的旧格式标签 `old_label.txt` 以及 `old_label.jsonl`。

   ```python
    data_root = 'tests/data/rec_toy_dataset/'

    # 读取旧版 txt 格式识别数据标签
    txt_dataset = dict(
        type='RecogTextDataset',
        data_root=data_root,
        ann_file='old_label.txt',
        data_prefix=dict(img_path='imgs'),
        parser_cfg=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1]),
        pipeline=[])

    # 读取旧版 json line 格式识别数据标签
    jsonl_dataset = dict(
        type='RecogTextDataset',
        data_root=data_root,
        ann_file='old_label.jsonl',
        data_prefix=dict(img_path='imgs'),
        parser_cfg=dict(
            type='LineJsonParser',
            keys=['filename', 'text'],
        pipeline=[])
   ```

3. [RecogLMDBDataset](mmocr.datasets.RecogLMDBDataset) 支持 0.x 版本文本识别任务的 `LMDB` 标注格式。只需要在 `configs/textrecog/_base_/datasets` 中添加新的数据集配置文件，并指定其数据集类型为 `RecogLMDBDataset` 即可。例如，以下示例展示了如何配置并读取 toy dataset 中的 `label.lmdb`，该 `lmdb` 文件**仅包含标签信息**。

   ```python
    data_root = 'tests/data/rec_toy_dataset/'

    lmdb_dataset = dict(
        type='RecogLMDBDataset',
        data_root=data_root,
        ann_file='label.lmdb',
        data_prefix=dict(img_path='imgs'),
        pipeline=[])
   ```

   当 `lmdb` 文件中既包含标签信息又包含图像时，我们除了需要将数据集类型设定为 `RecogLMDBDataset` 以外，还需要将数据流水线中的图像读取方法由 [`LoadImageFromFile`](mmocr.datasets.transforms.LoadImageFromFile) 替换为 [`LoadImageFromLMDB`](mmocr.datasets.transforms.LoadImageFromLMDB)。

   ```python
   # 将数据集类型设定为 RecogLMDBDataset
    data_root = 'tests/data/rec_toy_dataset/'

    lmdb_dataset = dict(
        type='RecogLMDBDataset',
        data_root=data_root,
        ann_file='imgs.lmdb',
        data_prefix=dict(img_path='imgs.lmdb'), # 将 img_path 设定为 lmdb 文件名
        pipeline=[])
   ```

   还需把 `train_pipeline` 及 `test_pipeline` 中的数据读取方法进行替换：

   ```python
    train_pipeline = [dict(type='LoadImageFromLMDB', color_type='grayscale', ignore_empty=True)]
   ```
