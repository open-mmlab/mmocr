# Dataset Types

## Dataset Wrapper

### UniformConcatDataset

`UniformConcatDataset` is a fundamental dataset wrapper in MMOCR which allows users to apply a universal pipeline on multiple datasets without specifying the pipeline for each of them.

#### Applying a Pipeline on Multiple Datasets

For example, to apply `train_pipeline` on both `train1` and `train2`,

```python
data = dict(
    ...
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1, train2],
        pipeline=train_pipeline))
```

Also, it support applying different `pipeline` to different `datasets`,

```python
train_list1 = [train1, train2]
train_list2 = [train3, train4]

data = dict(
    ...
    train=dict(
        type='UniformConcatDataset',
        datasets=[train_list1, train_list2],
        pipeline=[train_pipeline1, train_pipeline2]))
```

Here, `train_pipeline1` will be applied to `train1` and `train2`, and
`train_pipeline2` will be applied to `train3` and `train4`.

#### Getting Mean Evaluation Scores

Evaluating the model on multiple datasets is a common strategy in academia, and the mean score is therefore a critical indicator of the model's overall performance. By default, `UniformConcatDataset` reports mean scores in the form of
`mean_{metric_name}` when more than 1 datasets are wrapped. You can customize the behavior by setting
`show_mean_scores` in `data.val` and `data.test`. Choices are `'auto'`(default), `True` and `False`.

```python
data = dict(
    ...
    val=dict(
        type='UniformConcatDataset',
        show_mean_scores=True,  # always show mean scores
        datasets=[train_list],
        pipeline=[train_pipeline)
    test=dict(
        type='UniformConcatDataset',
        show_mean_scores=False,  # do not show mean scores
        datasets=[train_list],
        pipeline=[train_pipeline))
```

## Text Detection

### IcdarDataset

*Dataset with annotation file in coco-like json format*

#### Example Configuration


```python
dataset_type = 'IcdarDataset'
prefix = 'tests/data/toy_dataset/'
test=dict(
        type=dataset_type,
        ann_file=prefix + 'instances_test.json',
        img_prefix=prefix + 'imgs',
        pipeline=test_pipeline)
```

#### Annotation Format

You can check the content of the annotation file in `tests/data/toy_dataset/instances_test.json` for an example.
It's compatible with any annotation file in COCO format defined in [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py):

:::{note}
Icdar 2015/2017 and ctw1500 annotations need to be converted into the COCO format following the steps in [datasets.md](datasets.md).
:::

#### Evaluation

`IcdarDataset` has implemented two evaluation metrics, `hmean-iou` and `hmean-ic13`, to evaluate the performance of text detection models, where `hmean-iou` is the most widely used metric which computes precision, recall and F-score based on IoU between ground truth and prediction.

In particular, filtering predictions with a reasonable score threshold greatly impacts the performance measurement. MMOCR alleviates such hyperparameter effect by sweeping through the hyperparameter space and returns the best performance every evaluation time.
User can tune the searching scheme by passing `min_score_thr`, `max_score_thr` and `step` into the evaluation hook in the config.

For example, with the following configuration, you can evaluate the model's output on a list of boundary score thresholds [0.1, 0.2, 0.3, 0.4, 0.5] and get the best score from them **during training**.

```python
evaluation = dict(
    interval=100,
    metric='hmean-iou',
    min_score_thr=0.1,
    max_score_thr=0.5,
    step=0.1)
```

**During testing**, you can change these parameter values by appending them to `--eval-options`.

```bash
python tools/test.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py db_r18.pth --eval hmean-iou --eval-options min_score_thr=0.1 max_score_thr=0.6 step=0.1
```

Check out our [API doc](https://mmocr.readthedocs.io/en/latest/api.html#mmocr.core.evaluation.eval_hmean) for further explanations on these parameters.

### TextDetDataset

*Dataset with annotation file in line-json txt format*

We have designed new types of dataset consisting of **loader** , **backend**, and **parser** to load and parse different types of annotation files.
- **loader**: Load the annotation file. We now have a unified loader, `AnnFileLoader`, which can use different `backend` to load annotation from txt. The original `HardDiskLoader` and `LmdbLoader` will be deprecated.
- **backend**: Load annotation from different format and backend.
  - `LmdbAnnFileBackend`: Load annotation from lmdb dataset.
  - `HardDiskAnnFileBackend`: Load annotation file with raw hard disks storage backend. The annotation format can be either txt or lmdb.
  - `PetrelAnnFileBackend`: Load annotation file with petrel storage backend. The annotation format can be either txt or lmdb.
  - `HTTPAnnFileBackend`: Load annotation file with http storage backend. The annotation format can be either txt or lmdb.
- **parser**: Parse the annotation file line-by-line and return with `dict` format. There are two types of parser, `LineStrParser` and `LineJsonParser`.
  - `LineStrParser`: Parse one line in ann file while treating it as a string and separating it to several parts by a `separator`. It can be used on tasks with simple annotation files such as text recognition where each line of the annotation files contains the `filename` and `label` attribute only.
  - `LineJsonParser`: Parse one line in ann file while treating it as a json-string and using `json.loads` to convert it to `dict`. It can be used on tasks with complex annotation files such as text detection where each line of the annotation files contains multiple attributes (e.g. `filename`, `height`, `width`, `box`, `segmentation`, `iscrowd`, `category_id`, etc.).

#### Example Configuration

```python
dataset_type = 'TextDetDataset'
img_prefix = 'tests/data/toy_dataset/imgs'
test_anno_file = 'tests/data/toy_dataset/instances_test.txt'
test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=test_anno_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=4,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=test_pipeline,
    test_mode=True)
```

#### Annotation Format

The results are generated in the same way as the segmentation-based text recognition task above.
You can check the content of the annotation file in `tests/data/toy_dataset/instances_test.txt`.
The combination of `HardDiskLoader` and `LineJsonParser` will return a dict for each file by calling `__getitem__`:
```python
{"file_name": "test/img_10.jpg", "height": 720, "width": 1280, "annotations": [{"iscrowd": 1, "category_id": 1, "bbox": [260.0, 138.0, 24.0, 20.0], "segmentation": [[261, 138, 284, 140, 279, 158, 260, 158]]}, {"iscrowd": 0, "category_id": 1, "bbox": [288.0, 138.0, 129.0, 23.0], "segmentation": [[288, 138, 417, 140, 416, 161, 290, 157]]}, {"iscrowd": 0, "category_id": 1, "bbox": [743.0, 145.0, 37.0, 18.0], "segmentation": [[743, 145, 779, 146, 780, 163, 746, 163]]}, {"iscrowd": 0, "category_id": 1, "bbox": [783.0, 129.0, 50.0, 26.0], "segmentation": [[783, 129, 831, 132, 833, 155, 785, 153]]}, {"iscrowd": 1, "category_id": 1, "bbox": [831.0, 133.0, 43.0, 23.0], "segmentation": [[831, 133, 870, 135, 874, 156, 835, 155]]}, {"iscrowd": 1, "category_id": 1, "bbox": [159.0, 204.0, 72.0, 15.0], "segmentation": [[159, 205, 230, 204, 231, 218, 159, 219]]}, {"iscrowd": 1, "category_id": 1, "bbox": [785.0, 158.0, 75.0, 21.0], "segmentation": [[785, 158, 856, 158, 860, 178, 787, 179]]}, {"iscrowd": 1, "category_id": 1, "bbox": [1011.0, 157.0, 68.0, 16.0], "segmentation": [[1011, 157, 1079, 160, 1076, 173, 1011, 170]]}]}
```

#### Evaluation

`TextDetDataset` shares a similar implementation with `IcdarDataset`. Please refer to the evaluation section of ['IcdarDataset'](#icdardataset).


## Text Recognition

### OCRDataset

*Dataset for encoder-decoder based recognizer*

It shares a similar architecture with `TextDetDataset`. Check out the [introduction](#textdetdataset) for details.

#### Example Configuration

```python
dataset_type = 'OCRDataset'
img_prefix = 'tests/data/ocr_toy_dataset/imgs'
train_anno_file = 'tests/data/ocr_toy_dataset/label.txt'
train = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file,
    loader=dict(
        type='AnnFileLoader',
        repeat=10,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)
```

**Optional Arguments:**

- `repeat`: The number of repeated lines in the annotation files. For example, if there are `10` lines in the annotation file, setting `repeat=10` will generate a corresponding annotation file with size `100`.

#### Annotation Format

You can check the content of the annotation file in `tests/data/ocr_toy_dataset/label.txt`.
The combination of `HardDiskLoader` and `LineStrParser` will return a dict for each file by calling `__getitem__`: `{'filename': '1223731.jpg', 'text': 'GRAND'}`.

#### Loading LMDB Datasets

We have support for reading annotation files from the full lmdb dataset (with images and annotations). It is now possible to read lmdb datasets commonly used in academia. We have also implemented a new dataset conversion tool, [recog2lmdb](https://github.com/open-mmlab/mmocr/blob/main/tools/data/utils/recog2lmdb.py). It converts the recognition dataset to lmdb format. See [PR982](https://github.com/open-mmlab/mmocr/pull/982) for more details.

Here is an example configuration to load lmdb annotations:

```python
lmdb_root = 'path to lmdb folder'
train = dict(
    type='OCRDataset',
    img_prefix=lmdb_root,
    ann_file=lmdb_root,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
```

#### Evaluation

There are six evaluation metrics available for text recognition tasks: `word_acc`, `word_acc_ignore_case`, `word_acc_ignore_case_symbol`, `char_recall`, `char_precision` and `one_minus_ned`. See our [API doc](https://mmocr.readthedocs.io/en/latest/api.html#mmocr.core.evaluation.eval_ocr_metric) for explanations on metrics.

By default, `OCRDataset` generates full reports on all the metrics if its evaluation
metric is `acc`. Here is an example case for **training**.

```python
# Configuration
evaluation = dict(interval=1, metric='acc')
```

```bash
# Results
{'0_char_recall': 0.0484, '0_char_precision': 0.6, '0_word_acc': 0.0, '0_word_acc_ignore_case': 0.0, '0_word_acc_ignore_case_symbol': 0.0, '0_1-N.E.D': 0.0525}
```

:::{note}
'0_' prefixes result from `UniformConcatDataset`. It's kept here since MMOCR always wrap `UniformConcatDataset` around any datasets.
:::

If you want to conduct the evaluation on a subset of evaluation metrics:

```python
evaluation = dict(interval=1, metric=['word_acc_ignore_case', 'one_minus_ned'])
```

The result will look like:

```bash
{'0_word_acc_ignore_case': 0.0, '0_1-N.E.D': 0.0525}
```

**During testing**, you can specify the metrics to evaluate in the command line:

```bash
python tools/test.py configs/textrecog/crnn/crnn_toy_dataset.py crnn.pth --eval word_acc_ignore_case one_minus_ned
```

### OCRSegDataset

*Dataset for segmentation-based recognizer*

It shares a similar architecture with `TextDetDataset`. Check out the [introduction](#textdetdataset) for details.


#### Example Configuration

```python
prefix = 'tests/data/ocr_char_ann_toy_dataset/'
train = dict(
    type='OCRSegDataset',
    img_prefix=prefix + 'imgs',
    ann_file=prefix + 'instances_train.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=10,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'annotations', 'text'])),
    pipeline=train_pipeline,
    test_mode=True)
```

#### Annotation Format

You can check the content of the annotation file in `tests/data/ocr_char_ann_toy_dataset/instances_train.txt`.
The combination of `HardDiskLoader` and `LineJsonParser` will return a dict for each file by calling `__getitem__` each time:

```python
{"file_name": "resort_88_101_1.png", "annotations": [{"char_text": "F", "char_box": [11.0, 0.0, 22.0, 0.0, 12.0, 12.0, 0.0, 12.0]}, {"char_text": "r", "char_box": [23.0, 2.0, 31.0, 1.0, 24.0, 11.0, 16.0, 11.0]}, {"char_text": "o", "char_box": [33.0, 2.0, 43.0, 2.0, 36.0, 12.0, 25.0, 12.0]}, {"char_text": "m", "char_box": [46.0, 2.0, 61.0, 2.0, 53.0, 12.0, 39.0, 12.0]}, {"char_text": ":", "char_box": [61.0, 2.0, 69.0, 2.0, 63.0, 12.0, 55.0, 12.0]}], "text": "From:"}
```
