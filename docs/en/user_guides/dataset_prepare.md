# Dataset Preparation

## Downloading Datasets and Format Conversion

MMOCR supports dozens of widely used text [detection](../data_prepare/det.md) and [recognition](../data_prepare/recog.md) datasets, and provides detailed tutorials on data preparation.

As an example of the data preparation steps, you can perform the following steps to prepare the ICDAR 2015 dataset for text detection task.

- Download the ICDAR 2015 dataset from the [official ICDAR website](https://rrc.cvc.uab.es/?ch=4&com=downloads). Extract the training set `ch4_training_word_images_gt.zip` and the test set zip `ch4_test_word_images_gt.zip` to the path `data/icdar2015` respectively.

- Download the MMOCR format annotation files [train_label.json](<>) and [test_label.json](<>).

- After completing the above steps, the file directory structure is as follows

```text
├── data/icdar2015
│   ├── train_label.json
│   ├── test_label.json
│   ├── ch4_training_word_images_gt
│   └── ch4_test_word_images_gt
```

## Dataset Configuration

### Single Dataset Training and Evaluation

When training or evaluating a model on new datasets, we need to write the dataset config where the image path, annotation path, and image prefix are set. The path `configs/xxx/_base_/datasets/` is pre-configured with the commonly used datasets in MMOCR, here we take the ICDAR 2015 dataset as an example (see `configs/_base_/det_datasets/icdar2015.py`).

```Python
ic15_det_data_root = 'data/det/icdar2015' # dataset root path

# Train set config
ic15_det_train = dict(
    type='OCRDataset',
    data_root=ic15_det_data_root,                        # dataset root path
    ann_file='instances_training.json',                  # name of annotation
    data_prefix=dict(img_path='imgs/'),                  # prefix of image path
    filter_cfg=dict(filter_empty_gt=True, min_size=32),  # filtering empty images
    pipeline=None)
# Test set config
ic15_det_test = dict(
    type='OCRDataset',
    data_root=ic15_det_data_root,
    ann_file='instances_test.json',
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=None)
```

After configuring the dataset, we can import it in the corresponding model configs. For example, to train the "DBNet_R18" model on the ICDAR 2015 dataset.

```Python
_base_ = [
    '_base_dbnet_r18_fpnc.py',
    '../_base_/datasets/icdar2015.py',  # import the dataset config
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

ic15_det_train = _base_.ic15_det_train            # specify the training set
ic15_det_train.pipeline = _base_.train_pipeline   # specify the training pipeline
ic15_det_test = _base_.ic15_det_test              # specify the testing set
ic15_det_test.pipeline = _base_.test_pipeline     # specify the testing pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_det_train)    # specify the dataset in train_dataloader

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ic15_det_test)    # specify the dataset in val_dataloader

test_dataloader = val_dataloader
```

### Multi-dataset Training and Evaluation

In addition, [`ConcatDataset`](mmocr.datasets.ConcatDataset) enables users to train or test the model on a combination of multiple datasets. You just need to set the dataset type in the dataloader to `ConcatDataset` in the configuration file and specify the corresponding list of datasets.

```Python
train_list = [ic11, ic13, ic15]
train_dataloader = dict(
    dataset=dict(
        type='ConcatDataset', datasets=train_list, pipeline=train_pipeline))
```

For example, the following configuration uses the MJSynth dataset for training and 6 academic datasets (CUTE80, IIIT5K, SVT, SVTP, ICDAR2013, ICDAR2015) for testing.

```Python
_base_ = [ # Import all dataset configurations you want to use
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

# List of training datasets
train_list = [_base_.mj_rec_train]
# List of testing datasets
test_list = [
    _base_.cute80_rec_test, _base_.iiit5k_rec_test, _base_.svt_rec_test,
    _base_.svtp_rec_test, _base_.ic13_rec_test, _base_.ic15_rec_test
]

# Use ConcatDataset to combine the datasets in the list
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

# The order of the prefixes here must match the order of the datasets in the test_list list
val_evaluator = dict(
    dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
test_evaluator = val_evaluator
```

Note that when using multi-dataset for testing, we need to specify the evaluator type as [`MultiDatasetsEvaluator`](mmocr.evaluation.MultiDatasetsEvaluator). Also, the order of names in `dataset_prefixes` must be consistent with the order of datasets in the `test_list` so that the prefix name of test scores can be printed correctly in the log.
