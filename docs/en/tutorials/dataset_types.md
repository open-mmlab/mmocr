# Dataset Types

## General Introduction

To support the tasks of text detection, text recognition and key information extraction, we have designed some new types of dataset which consist of **loader** and **parser** to load and parse different types of annotation files.
- **loader**: Load the annotation file. There are two types of loader, `HardDiskLoader` and `LmdbLoader`
  - `HardDiskLoader`: Load `txt` format annotation file from hard disk to memory.
  - `LmdbLoader`: Load `lmdb` format annotation file with lmdb backend, which is very useful for **extremely large** annotation files to avoid out-of-memory problem when ten or more GPUs are used, since each GPU will start multiple processes to load annotation file to memory.
- **parser**: Parse the annotation file line-by-line and return with `dict` format. There are two types of parser, `LineStrParser` and `LineJsonParser`.
  - `LineStrParser`: Parse one line in ann file while treating it as a string and separating it to several parts by a `separator`. It can be used on tasks with simple annotation files such as text recognition where each line of the annotation files contains the `filename` and `label` attribute only.
  - `LineJsonParser`: Parse one line in ann file while treating it as a json-string and using `json.loads` to convert it to `dict`. It can be used on tasks with complex annotation files such as text detection where each line of the annotation files contains multiple attributes (e.g. `filename`, `height`, `width`, `box`, `segmentation`, `iscrowd`, `category_id`, etc.).

Here we show some examples of using different combination of `loader` and `parser`.

## General Task

### UniformConcatDataset

`UniformConcatDataset` is a dataset wrapper which allows users to apply a universal pipeline on multiple datasets without specifying the pipeline for each of them.

For example, to apply `train_pipeline` on both `train1` and `train2`,

```python
data = dict(
    ...
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1, train2],
        pipeline=train_pipeline))
```

Also, it support apply different `pipeline` to different `datasets`,

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

## Text Detection Task

### TextDetDataset

*Dataset with annotation file in line-json txt format*

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
The results are generated in the same way as the segmentation-based text recognition task above.
You can check the content of the annotation file in `tests/data/toy_dataset/instances_test.txt`.
The combination of `HardDiskLoader` and `LineJsonParser` will return a dict for each file by calling `__getitem__`:
```python
{"file_name": "test/img_10.jpg", "height": 720, "width": 1280, "annotations": [{"iscrowd": 1, "category_id": 1, "bbox": [260.0, 138.0, 24.0, 20.0], "segmentation": [[261, 138, 284, 140, 279, 158, 260, 158]]}, {"iscrowd": 0, "category_id": 1, "bbox": [288.0, 138.0, 129.0, 23.0], "segmentation": [[288, 138, 417, 140, 416, 161, 290, 157]]}, {"iscrowd": 0, "category_id": 1, "bbox": [743.0, 145.0, 37.0, 18.0], "segmentation": [[743, 145, 779, 146, 780, 163, 746, 163]]}, {"iscrowd": 0, "category_id": 1, "bbox": [783.0, 129.0, 50.0, 26.0], "segmentation": [[783, 129, 831, 132, 833, 155, 785, 153]]}, {"iscrowd": 1, "category_id": 1, "bbox": [831.0, 133.0, 43.0, 23.0], "segmentation": [[831, 133, 870, 135, 874, 156, 835, 155]]}, {"iscrowd": 1, "category_id": 1, "bbox": [159.0, 204.0, 72.0, 15.0], "segmentation": [[159, 205, 230, 204, 231, 218, 159, 219]]}, {"iscrowd": 1, "category_id": 1, "bbox": [785.0, 158.0, 75.0, 21.0], "segmentation": [[785, 158, 856, 158, 860, 178, 787, 179]]}, {"iscrowd": 1, "category_id": 1, "bbox": [1011.0, 157.0, 68.0, 16.0], "segmentation": [[1011, 157, 1079, 160, 1076, 173, 1011, 170]]}]}
```


### IcdarDataset

*Dataset with annotation file in coco-like json format*

For text detection, you can also use an annotation file in a COCO format that is defined in [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py):
```python
dataset_type = 'IcdarDataset'
prefix = 'tests/data/toy_dataset/'
test=dict(
        type=dataset_type,
        ann_file=prefix + 'instances_test.json',
        img_prefix=prefix + 'imgs',
        pipeline=test_pipeline)
```
You can check the content of the annotation file in `tests/data/toy_dataset/instances_test.json`.

:::{note}
Icdar 2015/2017 and ctw1500 annotations need to be converted into the COCO format following the steps in [datasets.md](datasets.md).
:::

## Text Recognition Task

### OCRDataset

*Dataset for encoder-decoder based recognizer*

```python
dataset_type = 'OCRDataset'
img_prefix = 'tests/data/ocr_toy_dataset/imgs'
train_anno_file = 'tests/data/ocr_toy_dataset/label.txt'
train = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=10,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)
```
You can check the content of the annotation file in `tests/data/ocr_toy_dataset/label.txt`.
The combination of `HardDiskLoader` and `LineStrParser` will return a dict for each file by calling `__getitem__`: `{'filename': '1223731.jpg', 'text': 'GRAND'}`.

**Optional Arguments:**

- `repeat`: The number of repeated lines in the annotation files. For example, if there are `10` lines in the annotation file, setting `repeat=10` will generate a corresponding annotation file with size `100`.

If the annotation file is extremely large, you can convert it from txt format to lmdb format with the following command:
```python
python tools/data_converter/txt2lmdb.py -i ann_file.txt -o ann_file.lmdb
```

After that, you can use `LmdbLoader` in dataset like below.
```python
img_prefix = 'tests/data/ocr_toy_dataset/imgs'
train_anno_file = 'tests/data/ocr_toy_dataset/label.lmdb'
train = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file,
    loader=dict(
        type='LmdbLoader',
        repeat=10,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)
```

### For dataset contains blank spaces

It is noteworthy that the `LineStrParser` should **NOT** be used to parse the annotation files containing multiple blank spaces (in file name or recognition transcriptions). The users have to convert the `plain txt` annotations to `json lines` to enable space recognition. For example:

```txt
% A plain txt annotation file that contains blank spaces
test/img 1.jpg Hello World!
test/img 2.jpg Hello Open MMLab!
test/img 3.jpg Hello MMOCR!
```

The `LineStrParser` will split the above annotation line to pieces (e.g. ['test/img', '1.jpg', 'Hello', 'World!']) that cannot be matched to the `keys` (e.g. ['filename', 'text']). Therefore, we need to convert it to a json line format by `json.dumps` (check [here](https://github.com/open-mmlab/mmocr/blob/main/tools/data/textrecog/funsd_converter.py#L175-L180) to see how to dump `jsonl`), and then the annotation file will look like as follows:

```txt
% A json line annotation file that contains blank spaces
{"filename": "test/img 1.jpg", "text": "Hello World!"}
{"filename": "test/img 2.jpg", "text": "Hello Open MMLab!"}
{"filename": "test/img 3.jpg", "text": "Hello MMOCR!"}
```

After converting the annotation format, you just need to set the parser arguments as:

```python
parser=dict(
    type='LineJsonParser',
    keys=['filename', 'text']))
```

Besides, you need to specify a dict that contains blank space to enablel blank recognition. Particularly, MMOCR provides two [built-in dicts](https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/convertors/base.py) `DICT37` and `DICT91` that contain blank space. For example, change the default `dict_type` in `configs/_base_/recog_models/crnn.py` to `DICT37`.

```python
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT37', with_unknown=False, lower=True) # ['DICT36', 'DICT37', 'DICT90', 'DICT91']
```

### OCRSegDataset

*Dataset for segmentation-based recognizer*

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
You can check the content of the annotation file in `tests/data/ocr_char_ann_toy_dataset/instances_train.txt`.
The combination of `HardDiskLoader` and `LineJsonParser` will return a dict for each file by calling `__getitem__` each time:
```python
{"file_name": "resort_88_101_1.png", "annotations": [{"char_text": "F", "char_box": [11.0, 0.0, 22.0, 0.0, 12.0, 12.0, 0.0, 12.0]}, {"char_text": "r", "char_box": [23.0, 2.0, 31.0, 1.0, 24.0, 11.0, 16.0, 11.0]}, {"char_text": "o", "char_box": [33.0, 2.0, 43.0, 2.0, 36.0, 12.0, 25.0, 12.0]}, {"char_text": "m", "char_box": [46.0, 2.0, 61.0, 2.0, 53.0, 12.0, 39.0, 12.0]}, {"char_text": ":", "char_box": [61.0, 2.0, 69.0, 2.0, 63.0, 12.0, 55.0, 12.0]}], "text": "From:"}
```
