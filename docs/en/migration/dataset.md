# Dataset Migration

Based on the new design of [BaseDataset](mmengine.dataset.BaseDataset) in [MMEngine](https://github.com/open-mmlab/mmengine), we have refactored the base OCR dataset class [`OCRDataset`](mmocr.datasets.OCRDataset) in MMOCR 1.0. The following document describes the differences between the old and new dataset formats in MMOCR, and how to migrate from the deprecated version to the latest. For users who do not want to migrate datasets at this time, we also provide a temporary solution in [Section Compatibility](#compatibility).

```{note}
The Key Information Extraction task still uses the original WildReceipt dataset annotation format.
```

## Review of Old Dataset Formats

MMOCR version 0.x implements a number of dataset classes, such as `IcdarDataset`, `TextDetDataset` for text detection tasks, and `OCRDataset`, `OCRSegDataset` for text recognition tasks. At the same time, the annotations may vary in different formats, such as `.txt`, `.json`, `.jsonl`. Users have to manually configure the `Loader` and the `Parser` while customizing the datasets.

### Text Detection

For the text detection task, `IcdarDataset` uses a COCO-like annotation format.

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

The `TextDetDataset` uses the JSON Line storage format, converting COCO-like labels to strings and saves them in `.txt` or `.jsonl` format files.

```text
{"file_name": "test/img_2.jpg", "height": 720, "width": 1280,  "annotations": [{"iscrowd": 0, "category_id": 1, "bbox": [602.0, 173.0,  33.0, 24.0], "segmentation": [[602, 173, 635, 175, 634, 197, 602,  196]]}, {"iscrowd": 0, "category_id": 1, "bbox": [734.0, 310.0, 58.0,  54.0], "segmentation": [[734, 310, 792, 320, 792, 364, 738, 361]]}]}
{"file_name": "test/img_5.jpg", "height": 720, "width": 1280,  "annotations": [{"iscrowd": 1, "category_id": 1, "bbox": [405.0, 409.0,  32.0, 52.0], "segmentation": [[408, 409, 437, 436, 434, 461, 405,  433]]}, {"iscrowd": 1, "category_id": 1, "bbox": [435.0, 434.0, 8.0,  33.0], "segmentation": [[437, 434, 443, 440, 441, 467, 435, 462]]}]}
```

### Text Recognition

For text recognition tasks, there are two annotation formats in MMOCR version 0.x. The simple `.txt` annotations separate image name and word annotation by a blank space, which cannot handle the case when spaces are included in a text instance.

```text
img1.jpg OpenMMLab
img2.jpg MMOCR
```

The JSON Line format uses a dictionary-like structure to represent the annotations, where the keys `filename` and `text` store the image name and word label, respectively.

```json
{"filename": "img1.jpg", "text": "OpenMMLab"}
{"filename": "img2.jpg", "text": "MMOCR"}
```

## New Dataset Format

To solve the dataset issues, MMOCR 1.x adopts a unified dataset design introduced in MMEngine. Each annotation file is a `.json` file that stores a `dict`, containing both `metainfo` and `data_list`, where the former includes basic information about the dataset and the latter consists of the label item of each target instance.

```json
{
  "metainfo":
    {
      "classes": ("cat", "dog"),
      // ...
    },
  "data_list":
    [
      {
        "img_path": "xxx/xxx_0.jpg",
        "img_label": 0,
        // ...
      },
      // ...
    ]
}
```

Based on the above structure, we introduced `TextDetDataset`, `TextRecogDataset` for MMOCR-specific tasks.

### Text Detection

#### Introduction of the New Format

The `TextDetDataset` holds the information required by the text detection task, such as bounding boxes and labels. We refer users to `tests/data/det_toy_dataset/instances_test.json` which is an example annotation for `TextDetDataset`.

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
            // ...
          ]
      }
    ]
}
```

The bounding box format is as follows: `[min_x, min_y, max_x, max_y]`

#### Migration Script

We provide a migration script to help users migrate old annotation files to the new format.

```bash
python tools/dataset_converters/textdet/data_migrator.py ${IN_PATH} ${OUT_PATH}
```

| ARGS     | Type                             | Description                                                                                                                                                      |
| -------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| in_path  | str                              | （Required）Path to the old annotation file.                                                                                                                     |
| out_path | str                              | （Required）Path to the new annotation file.                                                                                                                     |
| --task   | 'auto', 'textdet', 'textspotter' | Specifies the compatible task for the output dataset annotation. If 'textdet' is specified, the text field in coco format will not be dumped. The default is 'auto', which automatically determines the output format based on the the old annotation files. |

### Text Recognition

#### Introduction of the New Format

The `TextRecogDataset` holds the information required by the text detection task, such as text and image path. We refer users to `tests/data/rec_toy_dataset/labels.json` which is an example annotation for `TextRecogDataset`.

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

#### Migration Script

We provide a migration script to help users migrate old annotation files to the new format.

```bash
python tools/dataset_converters/textrecog/data_migrator.py ${IN_PATH} ${OUT_PATH} --format ${txt, jsonl, lmdb}
```

| ARGS     | Type                   | Description                                       |
| -------- | ---------------------- | ------------------------------------------------- |
| in_path  | str                    | （Required）Path to the old annotation file.      |
| out_path | str                    | （Required）Path to the new annotation file.      |
| --format | 'txt', 'jsonl', 'lmdb' | Specify the format of the old dataset annotation. |

## Compatibility

In consideration of the cost to users for data migration, we have temporarily made MMOCR version 1.x compatible with the old MMOCR 0.x format.

```{note}
The code and components used for compatibility with the old data format may be completely removed in a future release. Therefore, we strongly recommend that users migrate their datasets to the new data format.
```

Specifically, we provide three dataset classes [IcdarDataset](mmocr.datasets.IcdarDataset), [RecogTextDataset](mmocr.datasets.RecogTextDataset), [RecogLMDBDataset](mmocr.datasets.RecogLMDBDataset) to support the old formats.

1. [IcdarDataset](mmocr.datasets.IcdarDataset) supports COCO-like format annotations for text detection. You just need to add a new dataset config to `configs/textdet/_base_/datasets` and specify its dataset type as `IcdarDataset`.

   ```python
   data_root = 'data/det/icdar2015'
   train_anno_path = 'instances_training.json'

   train_dataset = dict(
       type='IcdarDataset',
       data_root=data_root,
       ann_file=train_anno_path,
       data_prefix=dict(img_path='imgs/'),
       filter_cfg=dict(filter_empty_gt=True, min_size=32),
       pipeline=None)
   ```

2. [RecogTextDataset](mmocr.datasets.RecogTextDataset) supports `.txt` and `.jsonl` format annotations for text recognition. You just need to add a new dataset config to `configs/textrecog/_base_/datasets` and specify its dataset type as `RecogTextDataset`. For example, the following example shows how to configure and load the 0.x format labels `old_label.txt` and `old_label.jsonl` from the toy dataset.

   ```python
    data_root = 'tests/data/rec_toy_dataset/'

    # loading 0.x txt format annos
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

    # loading 0.x json line format annos
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

3. [RecogLMDBDataset](mmocr.datasets.RecogLMDBDataset) supports LMDB format annotations for text recognition. You just need to add a new dataset config to `configs/textrecog/_base_/datasets` and specify its dataset type as `RecogLMDBDataset`. For example, the following example shows how to configure and load the **label-only lmdb** `label.lmdb` from the toy dataset.

   ```python
    data_root = 'tests/data/rec_toy_dataset/'

    lmdb_dataset = dict(
        type='RecogLMDBDataset',
        data_root=data_root,
        ann_file='label.lmdb',
        data_prefix=dict(img_path='imgs'),
        pipeline=[])
   ```

   When the `lmdb` file contains **both labels and images**, in addition to setting the dataset type to `RecogLMDBDataset` as in the above example, you also need to replace the [`LoadImageFromFile`](mmocr.datasets.transforms.LoadImageFromFile) with [`LoadImageFromLMDB`](mmocr.datasets.transforms.LoadImageFromLMDB) in the data pipelines.

   ```python
   # Specify the dataset type as RecogLMDBDataset
    data_root = 'tests/data/rec_toy_dataset/'

    lmdb_dataset = dict(
        type='RecogLMDBDataset',
        data_root=data_root,
        ann_file='imgs.lmdb',
        data_prefix=dict(img_path='imgs.lmdb'), # setting the img_path as the lmdb name
        pipeline=[])
   ```

   Also, replacing the image loading transforms in `train_pipeline` and `test_pipeline`, for example：

   ```python
    train_pipeline = [dict(type='LoadImageFromLMDB', color_type='grayscale', ignore_empty=True)]
   ```
