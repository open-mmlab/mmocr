# 开始

在这个指南中，将介绍一些常用的命令，来帮助你熟悉 MMOCR。同时还提供了[notebook](https://github.com/open-mmlab/mmocr/blob/main/demo/MMOCR_Tutorial.ipynb) 版本的代码，可以让您快速上手 MMOCR。

## 安装

查看[安装指南](install.md)，了解完整步骤。

## 数据集准备

MMOCR 支持许多种类数据集，这些数据集根据其相应任务的类型进行分类。可以在以下部分找到它们的准备步骤：[检测数据集](datasets/det.md)、[识别数据集](datasets/recog.md)、[KIE 数据集](datasets/kie.md)和 [NER 数据集](datasets/ner.md)。

## 使用预训练模型进行推理

下面通过一个简单的命令来演示端到端的识别：

```shell
python mmocr/utils/ocr.py demo/demo_text_ocr.jpg --print-result --imshow
```

其检测结果将被打印出来，并弹出一个新窗口显示结果。更多示例和完整说明可以在[示例](demo.md)中找到。

## 训练

### 小数据集训练

在`tests/data`目录下提供了一个用于训练演示的小数据集，在准备学术数据集之前，它可以演示一个初步的训练。

例如：用 `seg` 方法和小数据集来训练文本识别任务，
```shell
python tools/train.py configs/textrecog/seg/seg_r31_1by16_fpnocr_toy_dataset.py --work-dir seg
```

用 `sar` 方法和小数据集训练文本识别,
```shell
python tools/train.py configs/textrecog/sar/sar_r31_parallel_decoder_toy_dataset.py --work-dir sar
```

### 使用学术数据集进行训练

按照说明准备好所需的学术数据集后，最后要检查模型的配置是否将 MMOCR 指向正确的数据集路径。假设在 ICDAR2015 数据集上训练 DBNet,部分配置如 `configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py` 所示:
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
这里需要检查数据集路径 `data/icdar2015` 是否正确. 然后可以启动训练命令：
```shell
python tools/train.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py --work-dir dbnet
```

想要了解完整的训练参数配置可以查看 [Training](training.md)了解。

## 测试

如果完成了 DBNet 模型训练，并将最新的模型保存在 `dbnet/latest.pth`。可以使用以下命令，及`hmean-iou`指标来评估其在测试集上的性能：
```shell
python tools/test.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py dbnet/latest.pth --eval hmean-iou
```

还可以在线评估预训练模型，命令如下：
```shell
python tools/test.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth --eval hmean-iou
```

有关测试的更多说明，请参阅 [Testing](testing.md).
