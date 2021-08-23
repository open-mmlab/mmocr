# Getting Started

In this guide we will show you some useful commands and familiarize you with MMOCR. We also provide [a notebook](https://github.com/open-mmlab/mmocr/blob/main/demo/MMOCR_Tutorial.ipynb) that can help you get the most out of MMOCR.

## Installation

Check out our [installation guide](install.md) for full steps.

## Dataset Preparation

MMOCR supports numerous datasets which are classified by the type of their corresponding tasks. You may find their preparation steps in these sections: [Detection Datasets](datasets/det.md), [Recognition Datasets](datasets/recog.md), [KIE Datasets](datasets/kie.md) and [NER Datasets](datasets/ner.md).

## Inference with Pretrained Models

You can perform end-to-end OCR on our demo image with one simple line of command:

```shell
python mmocr/utils/ocr.py demo/demo_text_ocr.jpg --print-result --imshow
```

Its detection result will be printed out and a new window will pop up with result visualization. More demo and full instructions and be found in [Inference](inference.md).

## Training

### Training with Toy Dataset

We provide a toy dataset under `tests/data` on which you can get a sense of training before the academic dataset is prepared.

For example, to train a text recognition task with `seg` method and toy dataset,
```shell
python tools/train.py configs/textrecog/seg/seg_r31_1by16_fpnocr_toy_dataset.py --work-dir seg
```

To train a text recognition task with `sar` method and toy dataset,
```shell
python tools/train.py configs/textrecog/sar/sar_r31_parallel_decoder_toy_dataset.py --work-dir sar
```

### Training with Academic Dataset

Once you have prepared required academic dataset following our instruction, the only last thing to check is if the model's config points MMOCR to the correct dataset path. Suppose we want to train DBNet on ICDAR 2015, and part of `configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py` looks like the following:
```python
dataset_type = 'IcdarDataset'
data_root = 'data/icdar2015'
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_training.json',
        img_prefix=data_root + '/imgs',
        pipeline=train_pipeline)
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_test.json',
        img_prefix=data_root + '/imgs',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_test.json',
        img_prefix=data_root + '/imgs',
        pipeline=test_pipeline))
```
You would need to check if `data/icdar2015` is right. Then you can start training with the command:
```shell
python tools/train.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py --work-dir dbnet
```

You can find full training instructions, explanations and useful training configs in [Training](training.md).

## Testing

Suppose now you have finished the training of DBNet and the latest model has been saved in `dbnet/latest.pth`. You can evaluate its performance on the test set using the `hmean-iou` metric with the following command:
```shell
python tools/test.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py dbnet/latest.pth --eval hmean-iou
```

Evaluating any pretrained model accessible online is also allowed:
```shell
python tools/test.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth --eval hmean-iou
```

More instructions on testing are available in [Testing](testing.md).
