# 数据准备 (Beta)

```{note}
Dataset Preparer 目前仍处在公测阶段，欢迎尝鲜试用！如遇到任何问题，请及时向我们反馈。
```

## 一键式数据准备脚本

MMOCR 提供了统一的一站式数据集准备脚本 `prepare_dataset.py`。

仅需一行命令即可完成数据的下载、解压，以及格式转换。

```bash
python tools/dataset_converters/prepare_dataset.py [$DATASET_NAME] --task [$TASK] --nproc [$NPROC]
```

| 参数         | 类型 | 说明                                                                                                  |
| ------------ | ---- | ----------------------------------------------------------------------------------------------------- |
| dataset_name | str  | （必须）需要准备的数据集名称。                                                                        |
| --task       | str  | 将数据集格式转换为指定任务的 MMOCR 格式。可选项为： 'textdet', 'textrecog', 'textspotting' 和 'kie'。 |
| --nproc      | str  | 使用的进程数，默认为 4。                                                                              |

例如，以下命令展示了如何使用该脚本为 ICDAR2015 数据集准备文本检测任务所需的数据。

```bash
python tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet
```

该脚本也支持同时准备多个数据集，例如，以下命令展示了如何使用该脚本同时为 ICDAR2015 和 TotalText 数据集准备文本识别任务所需的数据。

```bash
python tools/dataset_converters/prepare_dataset.py icdar2015 totaltext --task textrecog
```

进一步了解 MMOCR 支持的数据集，您可以浏览[支持的数据集文档](./datasetzoo.md)

## 进阶用法

### 数据集配置

数据集自动化准备脚本使用了模块化的设计，极大地增强了扩展性，用户能够很方便地配置其他公开数据集或私有数据集。数据集自动化准备脚本的配置文件被统一存储在 `dataset_zoo/` 目录下，用户可以在该目录下找到所有已由 MMOCR 官方支持的数据集准备脚本配置文件。该文件夹的目录结构如下：

```text
dataset_zoo/
├── icdar2015
│   ├── metafile.yml
│   ├── textdet.py
│   ├── textrecog.py
│   └── textspotting.py
└── wildreceipt
    ├── metafile.yml
    ├── kie.py
    ├── textdet.py
    ├── textrecog.py
    └── textspotting.py
```

其中，`metafile.yml` 是数据集的元信息文件，其中存放了对应数据集的基本信息，包括发布年份，论文作者，以及版权等其他信息。其它以任务名命名的则是数据集准备脚本的配置文件，用于配置数据集的下载、解压、格式转换等操作。这些配置文件采用了 Python 格式，其使用方法与 MMOCR 算法库的其他配置文件完全一致，详见[配置文件文档](../config.md)。

#### 数据集元文件

以数据集 ICDAR2015 为例，`metafile.yml` 中存储了基础的数据集信息：

```yaml
Name: 'Incidental Scene Text IC15'
Paper:
  Title: ICDAR 2015 Competition on Robust Reading
  URL: https://rrc.cvc.uab.es/files/short_rrc_2015.pdf
  Venue: ICDAR
  Year: '2015'
  BibTeX: '@inproceedings{karatzas2015icdar,
  title={ICDAR 2015 competition on robust reading},
  author={Karatzas, Dimosthenis and Gomez-Bigorda, Lluis and Nicolaou, Anguelos and Ghosh, Suman and Bagdanov, Andrew and Iwamura, Masakazu and Matas, Jiri and Neumann, Lukas and Chandrasekhar, Vijay Ramaseshan and Lu, Shijian and others},
  booktitle={2015 13th international conference on document analysis and recognition (ICDAR)},
  pages={1156--1160},
  year={2015},
  organization={IEEE}}'
Data:
  Website: https://rrc.cvc.uab.es/?ch=4
  Language:
    - English
  Scene:
    - Natural Scene
  Granularity:
    - Word
  Tasks:
    - textdet
    - textrecog
    - textspotting
  License:
    Type: CC BY 4.0
    Link: https://creativecommons.org/licenses/by/4.0/
```

该文件在数据集准备过程中并不是强制要求的（因此用户在使用添加自己的私有数据集时可以忽略该文件），但为了用户更好地了解各个公开数据集的信息，我们建议用户在使用数据集准备脚本前阅读对应的元文件信息，以了解该数据集的特征是否符合用户需求。

#### 数据集准备脚本配置文件

下面，我们将介绍数据集准备脚本配置文件 `textXXX.py` 的默认字段与使用方法。

我们在配置文件中提供了 `data_root` 与 `cache_path` 两个默认字段，分别用于存放转换后的 MMOCR 格式的数据集文件，以及在数据准备过程中下载的压缩包等临时文件。

```python
data_root = './data/icdar2015'
cache_path = './data/cache'
```

其次，数据集的准备通常包含了“原始数据准备”以及“格式转换和保存”这两个主要步骤。因此，我们约定通过 `data_obtainer` 和 `data_converter` 参数来配置这两个步骤的行为。在某些情况下，用户也可以通过缺省 `data_converter` 参数来仅进行原始数据的下载和解压，而不进行格式转换和保存。或者，对于本地存储的数据集，通过缺省 `data_obtainer` 参数来仅进行格式转换和保存。

以 ICDAR2015 数据集的文本检测任务准备配置文件（`dataset_zoo/icdar2015/textdet.py`）为例：

```python
data_obtainer = dict(
    type='NaiveDataObtainer',
    cache_path=cache_path,
    data_root=data_root,
    files=[
        dict(
            url='https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',
            save_name='ic15_textdet_train_img.zip',
            md5='c51cbace155dcc4d98c8dd19d378f30d',
            split=['train'],
            content=['image'],
            mapping=[['ic15_textdet_train_img', 'imgs/train']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/ch4_test_images.zip',
            save_name='ic15_textdet_test_img.zip',
            md5='97e4c1ddcf074ffcc75feff2b63c35dd',
            split=['test'],
            content=['image'],
            mapping=[['ic15_textdet_test_img', 'imgs/test']]),
    ])
```

数据准备器 `data_obtainer` 的类型默认为 `NaiveDataObtainer`，其主要功能是依次下载压缩包并解压到指定目录。在这里，我们通过 `files` 参数来配置下载的压缩包的 URL、保存名称、MD5 值等信息。其中，`mapping` 参数用于指定该压缩包中的数据解压后的存放路径。另外 `split` 和 `content` 这两个可选参数则分别标明了该压缩包中存储的内容类型与其对应的数据集合。

```python
data_converter = dict(
    type='TextDetDataConverter',
    splits=['train', 'test'],
    data_root=data_root,
    gatherer=dict(
        type='pair_gather',
        suffixes=['.jpg', '.JPG'],
        rule=[r'img_(\d+)\.([jJ][pP][gG])', r'gt_img_\1.txt']),
    parser=dict(type='ICDARTxtTextDetAnnParser'),
    dumper=dict(type='JsonDumper'),
    delete=['annotations', 'ic15_textdet_test_img', 'ic15_textdet_train_img'])
```

数据转换器 `data_converter` 负责完成原始数据的读取与格式转换，并保存为 MMOCR 支持的格式。其中我们针对不同的任务，提供了内置的集中数据转换器，如文本检测任务数据转换器 `TextDetDataConverter`，文本识别任务数据转换器 `TextRecogDataConverter`，端到端文本检测识别任务转换器 `TextSpottingDataConverter`，以及关键信息抽取任务数据转换器 `WildReceiptConverter`（由于关键信息抽取任务目前仅支持 WildReceipt 数据集，我们暂时只提供了基于该数据集的数据转换器）。

以文本检测任务为例，`TextDetDataConverter` 主要完成以下工作：

- 收集并匹配原始数据集中的图片与标注文件，如图像 `img_1.jpg` 与 标注 `gt_img_1.txt`
- 读取原始标注文件，解析出文本框坐标与文本内容等必要信息
- 将解析后的数据统一转换至 MMOCR 支持的格式
- 将转换后的数据保存为指定路径和格式

以上个步骤我们分别可以通过 `gatherer`，`parser`，`dumper` 来进行配置。

具体而言，`gatherer` 用于收集并匹配原始数据集中的图片与标注文件。常用的 OCR 数据集通常有两种标注保存形式，一种为多个标注文件对应多张图片，一种则为单个标注文件对应多张图片，如：

```text
多对多
├── img_1.jpg
├── gt_img_1.txt
├── img_2.jpg
├── gt_img_2.txt
├── img_3.JPG
├── gt_img_3.txt

单对多
├── img_1.jpg
├── img_2.jpg
├── img_3.JPG
├── gt.txt
```

因此，我们内置了 `pair_gather` 与 `mono_gather` 来处理以上这两种情况。其中 `pair_gather` 用于多对多的情况，`mono_gather` 用于单对多的情况。`pair_gather` 需要指定 `suffixes` 参数，用于指定图片的后缀名，如上述例子中的 `suffixes=[.jpg,.JPG]`。此外，还需要通过正则表达式来指定图片与标注文件的对应关系，如上述例子中的 `rule=[r'img_(\d+)\.([jJ][pP][gG])'，r'gt_img_\1.txt']`。其中 `\d+` 用于匹配图片的序号，`([jJ][pP][gG])` 用于匹配图片的后缀名，`\_1` 则将匹配到的图片序号与标注文件序号对应起来。

当获取了图像与标注文件的对应关系后，data preparer 将解析原始标注文件。由于不同数据集的标注格式通常有很大的区别，当我们需要支持新的数据集时，通常需要实现一个新的 `parser` 来解析原始标注文件。parser 将任务相关的数据解析后打包成 MMOCR 的统一格式。

最后，我们可以通过指定不同的 dumper 来决定要将数据保存为何种格式。目前，我们仅支持 `JsonDumper` 与 `WildreceiptOpensetDumper`，其中，前者用于将数据保存为标准的 MMOCR Json 格式，而后者用于将数据保存为 Wildreceipt 格式。未来，我们计划支持 `LMDBDumper` 用于保存 LMDB 格式的标注文件。

### 使用 Data Preparer 准备自定义数据集

\[待更新\]
