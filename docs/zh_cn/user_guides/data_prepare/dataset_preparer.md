# 数据准备 (Beta)

```{note}
Dataset Preparer 目前仍处在公测阶段，欢迎尝鲜试用！如遇到任何问题，请及时向我们反馈。
```

## 一键式数据准备脚本

MMOCR 提供了统一的一站式数据集准备脚本 `prepare_dataset.py`。

仅需一行命令即可完成数据的下载、解压、格式转换，及基础配置的生成。

```bash
python tools/dataset_converters/prepare_dataset.py [$DATASET_NAME] [--task $TASK] [--nproc $NPROC] [--overwrite-cfg] [--dataset-zoo-path $DATASET_ZOO_PATH]
```

| 参数               | 类型 | 说明                                                                                                  |
| ------------------ | ---- | ----------------------------------------------------------------------------------------------------- |
| dataset_name       | str  | （必须）需要准备的数据集名称。                                                                        |
| --task             | str  | 将数据集格式转换为指定任务的 MMOCR 格式。可选项为： 'textdet', 'textrecog', 'textspotting' 和 'kie'。 |
| --nproc            | str  | 使用的进程数，默认为 4。                                                                              |
| --overwrite-cfg    | str  | 若数据集的基础配置已经在 `configs/{task}/_base_/datasets` 中存在，依然重写该配置                      |
| --dataset-zoo-path | str  | 存放数据库配置文件的路径。若不指定，则默认为 `./dataset_zoo`                                          |

例如，以下命令展示了如何使用该脚本为 ICDAR2015 数据集准备文本检测任务所需的数据。

```bash
python tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet --overwrite-cfg
```

该脚本也支持同时准备多个数据集，例如，以下命令展示了如何使用该脚本同时为 ICDAR2015 和 TotalText 数据集准备文本识别任务所需的数据。

```bash
python tools/dataset_converters/prepare_dataset.py icdar2015 totaltext --task textrecog --overwrite-cfg
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

其次，数据集的准备通常包含了“原始数据准备”、“格式转换和保存”及“生成基础配置”这三个主要步骤。因此，我们约定通过 `data_obtainer`、 `data_converter` 和 `config_generator` 参数来配置这三个步骤的行为。

```{note}
如果用户需要跳过某一步骤，则可以省略配置相应参数。例如，如果数据集本身已经遵循 MMOCR 的格式，用户就可以省略掉 `data_converter` 的配置来跳过数据集格式的转换。或者，如果用户不需要自动生成基础配置，可以忽略掉 `config_generator` 的配置。
```

接下来，我们将以 ICDAR2015 数据集的文本检测任务准备配置文件（`dataset_zoo/icdar2015/textdet.py`）为例，逐个模块解析 Dataset Preparer 的运行流程。

##### 原始数据准备 - `data_obtainer`

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
            mapping=[['ic15_textdet_train_img', 'textdet_imgs/train']]),
        dict(
            url='https://rrc.cvc.uab.es/downloads/'
            'ch4_training_localization_transcription_gt.zip',
            save_name='ic15_textdet_train_gt.zip',
            md5='3bfaf1988960909014f7987d2343060b',
            split=['train'],
            content=['annotation'],
            mapping=[['ic15_textdet_train_gt', 'annotations/train']]),
        # ...
    ])
```

数据准备器 `data_obtainer` 的类型默认为 `NaiveDataObtainer`，其主要功能是依次下载压缩包并解压到指定目录。在这里，我们通过 `files` 参数来配置下载的压缩包的 URL、保存名称、MD5 值等信息。其中，`mapping` 参数用于指定该压缩包中的数据解压后的存放路径。另外 `split` 和 `content` 这两个可选参数则分别标明了该压缩包中存储的内容类型与其对应的数据集合。

##### 格式转换和保存 - `data_converter`

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

数据转换器 `data_converter` 主要由转换器及子模块 `gatherer`、`parser` 及 `dumper` 组成，负责完成原始数据的读取与格式转换，并保存为 MMOCR 支持的格式。

目前，MMOCR 中支持的数据集转换器类别如下：

- 文本检测任务数据转换器 `TextDetDataConverter`
- 文本识别任务数据转换器
  - `TextRecogDataConverter`
  - `TextRecogCropConverter`
- 端到端文本检测识别任务转换器 `TextSpottingDataConverter`
- 关键信息抽取任务数据转换器 `WildReceiptConverter`

MMOCR 中目前支持的转换器主要以任务为边界，这是因为不同任务所需的数据格式有细微的差异。
比较特别的是，文本识别任务有两个数据转换器，这是因为不同的文本识别数据集提供文字图片的方式有所差别。有的数据集提供了仅包含文字的小图，它们天然适用于文本识别任务，可以直接使用 `TextRecogDataConverter` 处理。而有的数据集提供的是包含了周围场景的大图，因此在准备数据集时，我们需要预先根据标注信息把文字区域裁剪出来，这种情况下则要用到 `TextRecogCropConverter`。

简单介绍下 `TextRecogCropConverter` 数据转换器的使用方法：

- 由于标注文件的解析方式与 TextDet 环节一致，所以仅需继承 `dataset_zoo/xxx/textdet.py` 的  data_converter，并修改type值为 'TextRecogCropConverter'，`TextRecogCropConverter` 会在执行 `pack_instance()` 方法时根据解析获得的标注信息完成文字区域的裁剪。
- 同时，根据是否存在旋转文字区域标注内置了两种裁剪方式，默认按照水平文本框裁剪。如果存在旋转的文字区域，可以设置 `crop_with_warp=True` 切换为使用 OpenCV warpPerspective 方法进行裁剪。

```python
_base_ = ['textdet.py']

data_converter = dict(
  type='TextRecogCropConverter',
  crop_with_warp=True)
```

接下来，我们将具体解析 `data_converter` 的功能。以文本检测任务为例，`TextDetDataConverter` 与各子模块配合，主要完成以下工作：

- `gatherer` 负责收集并匹配原始数据集中的图片与标注文件，如图像 `img_1.jpg` 与标注 `gt_img_1.txt`
- `parser` 负责读取原始标注文件，解析出文本框坐标与文本内容等必要信息
- 转换器将解析后的数据统一转换至 MMOCR 中当前任务的格式
- `dumper` 将转换后的数据保存为指定路径和文件格式
- 转换器删除 `delete` 参数指定的临时文件

以上个步骤我们分别可以通过 `gatherer`，`parser`，`dumper`, `delete` 来进行配置。

###### `gatherer`

作为数据转换的第一步，`data_converter` 会通过 `gatherer` 遍历数据集目录下的文件，将图像与标注文件一一对应，并整理出一份文件列表供 `parser` 读取。因此，我们首先需要知道当前数据集下，图片文件与标注文件匹配的规则。

OCR 数据集通常有两种标注保存形式，一种为多个标注文件对应多张图片，一种则为单个标注文件对应多张图片，如：

```text
多对多
├── {taskname}_imgs/{split}/img_img_1.jpg
├── annotations/{split}/gt_img_1.txt
├── {taskname}_imgs/{split}/img_2.jpg
├── annotations/{split}/gt_img_2.txt
├── {taskname}_imgs/{split}/img_3.JPG
├── annotations/{split}/gt_img_3.txt

单对多
├── {taskname}/{split}/img_1.jpg
├── {taskname}/{split}/img_2.jpg
├── {taskname}/{split}/img_3.JPG
├── annotations/gt.txt
```

```{note}
为了简化处理，gatherer 约定数据集的图片和标注需要分别储存在 `{taskname}_imgs/{split}/` 和 `annotations/` 下。特别地，对于多对多的情况，标注文件需要放置于 `annotations/{split}` 下。如本例中，icdar 2015 训练集的图片就被储存在 `textdet_imgs/train/` 下，而训练的标注则被存在 `annotations/train/` 下。
```

因此，我们内置了 `pair_gather` 与 `mono_gather` 来处理以上这两种情况。其中 `pair_gather` 用于多对多的情况，`mono_gather` 用于单对多的情况。

在多对多的情况下，`pair_gather` 需要按照一定的命名规则找到图片文件和对应的标注文件。首先，我们需要通过 `suffixes` 参数指定图片的后缀名，如上述例子中的 `suffixes=[.jpg,.JPG]`。此外，还需要通过正则表达式来指定图片与标注文件的对应关系，如上述例子中的 `rule=[r'img_(\d+)\.([jJ][pP][gG])'，r'gt_img_\1.txt']`。其中 `\d+` 用于匹配图片的序号，`([jJ][pP][gG])` 用于匹配图片的后缀名，`\1` 则将匹配到的图片序号与标注文件序号对应起来。

单对多的情况通常比较简单，用户只需要指定每个 split 对应的标注文件即可。因此 `mono_gather` 预设了 `train_ann`、`val_ann` 和 `test_ann` 参数，供用户直接指定标注文件。

###### `parser`

当获取了图像与标注文件的对应关系后，data preparer 将解析原始标注文件。由于不同数据集的标注格式通常有很大的区别，当我们需要支持新的数据集时，**通常需要实现一个新的 `parser` 来解析原始标注文件**。parser 将任务相关的数据解析后打包成 MMOCR 的统一格式。

###### `dumper`

之后，我们可以通过指定不同的 dumper 来决定要将数据保存为何种格式。目前，我们仅支持 `JsonDumper` 与 `WildreceiptOpensetDumper`，其中，前者用于将数据保存为标准的 MMOCR Json 格式，而后者用于将数据保存为 Wildreceipt 格式。未来，我们计划支持 `LMDBDumper` 用于保存 LMDB 格式的标注文件。

###### `delete`

在处理数据集时，往往会产生一些不需要的临时文件。这里可以以列表的形式传入这些文件或文件夹，在结束转换时即会删除。

##### 生成基础配置 - `config_generator`

```python
config_generator = dict(type='TextDetConfigGenerator', data_root=data_root)
```

在准备好数据集的所有文件后，配置生成器 `TextDetConfigGenerator` 就会自动为 MMOCR 生成调用该数据集所需要的基础配置文件。生成后的文件默认会被置于 `configs/{task}/_base_/datasets/` 下。例如，本例中，icdar 2015 的基础配置文件就会被生成在 `configs/textdet/_base_/datasets/icdar2015.py` 下：

```python
icdar2015_textdet_data_root = 'data/icdar2015'

icdar2015_textdet_train = dict(
    type='OCRDataset',
    data_root=icdar2015_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

icdar2015_textdet_test = dict(
    type='OCRDataset',
    data_root=icdar2015_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
```

有了该文件后，我们就能从模型的配置文件中直接导入该数据集到 `dataloader` 中使用（以下样例节选自 [`configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py`](/configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py)）：

```python
_base_ = [
    '../_base_/datasets/icdar2015.py',
    # ...
]

# dataset settings
icdar2015_textdet_train = _base_.icdar2015_textdet_train
icdar2015_textdet_test = _base_.icdar2015_textdet_test
# ...

train_dataloader = dict(
    dataset=icdar2015_textdet_train)

val_dataloader = dict(
    dataset=icdar2015_textdet_test)

test_dataloader = val_dataloader
```

```{note}
除非用户在运行脚本的时候手动指定了 `overwrite-cfg`，配置生成器默认不会自动覆盖已经存在的基础配置文件。
```

由于每个任务所需的基本数据集配置格式不一，我们也针对各个任务推出了 `TextRecogConfigGenerator` 及 `TextSpottingConfigGenerator` 等生成器。

## 向 Dataset Preparer 添加新的数据集

### 添加公开数据集

MMOCR 已经支持了许多[常用的公开数据集](./datasetzoo.md)。如果你想用的数据集还没有被支持，并且你也愿意为 MMOCR 开源社区[贡献代码](../../notes/contribution_guide.md)，你可以按照以下步骤来添加一个新的数据集。

接下来我们以添加 **ICDAR2013** 数据集为例，展示如何一步一步地添加一个新的公开数据集。

#### 添加 `metafile.yml`

首先，我们确认 `dataset_zoo/` 中不存在我们准备添加的数据集。然后我们先新建以待添加数据集命名的文件夹，如 `icdar2013/`（通常，我们使用不包含符号的小写英文字母及数字来命名数据集）。在 `icdar2013/` 文件夹中，我们新建 `metafile.yml` 文件，并按照以下模板来填充数据集的基本信息：

```yaml
Name: 'Incidental Scene Text IC13'
Paper:
  Title: ICDAR 2013 Robust Reading Competition
  URL: https://www.imlab.jp/publication_data/1352/icdar_competition_report.pdf
  Venue: ICDAR
  Year: '2013'
  BibTeX: '@inproceedings{karatzas2013icdar,
  title={ICDAR 2013 robust reading competition},
  author={Karatzas, Dimosthenis and Shafait, Faisal and Uchida, Seiichi and Iwamura, Masakazu and i Bigorda, Lluis Gomez and Mestre, Sergi Robles and Mas, Joan and Mota, David Fernandez and Almazan, Jon Almazan and De Las Heras, Lluis Pere},
  booktitle={2013 12th international conference on document analysis and recognition},
  pages={1484--1493},
  year={2013},
  organization={IEEE}}'
Data:
  Website: https://rrc.cvc.uab.es/?ch=2
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
    Type: N/A
    Link: N/A
  Format: .txt
  Keywords:
    - Horizontal
```

`metafile.yml` 存储了数据集的基本信息，这些信息一方面可以帮助用户了解数据集的基本情况，另一方面也会被用于自动化脚本来生成相应的数据集文档。因此，当你向 MMOCR Dataset Preparer 添加新的数据集支持时，请按照以上模板来填充 `metafile.yml` 文件。具体地，我们在下表中列出每个字段对应的含义：

| 字段名           | 含义                                                                  |
| :--------------- | :-------------------------------------------------------------------- |
| Name             | 数据集的名称                                                          |
| Paper.Title      | 数据集论文的标题                                                      |
| Paper.URL        | 数据集论文的链接                                                      |
| Paper.Venue      | 数据集论文发表的会议/期刊名称                                         |
| Paper.Year       | 数据集论文发表的年份                                                  |
| Paper.BibTeX     | 数据集论文的引用的 BibTex                                             |
| Data.Website     | 数据集的官方网站                                                      |
| Data.Language    | 数据集支持的语言                                                      |
| Data.Scene       | 数据集支持的场景，如 `Natural Scene`, `Document`, `Handwritten` 等    |
| Data.Granularity | 数据集支持的粒度，如 `Character`, `Word`, `Line` 等                   |
| Data.Tasks       | 数据集支持的任务，如 `textdet`, `textrecog`, `textspotting`, `kie` 等 |
| Data.License     | 数据集的许可证信息，如果不存在许可证，则使用 `N/A` 填充               |
| Data.Format      | 数据集标注文件的格式，如 `.txt`, `.xml`, `.json` 等                   |
| Data.Keywords    | 数据集的特性关键词，如 `Horizontal`, `Vertical`, `Curved` 等          |

#### 添加对应任务的配置文件

在 `dataset_zoo/icdar2013` 中，我们接着添加以任务名称命名的 `.py` 配置文件。如 `textdet.py`，`textrecog.py`，`textspotting.py`，`kie.py` 等。在该文件中，我们需要对数据获取方式和转换方式进行配置。

##### 配置数据集获取方法 `data_obtainer`

以文本检测任务为例：

```python
data_obtainer = dict(
 type='NaiveDataObtainer',
 cache_path=cache_path,
 data_root=data_root,
 files=[
     dict(
         url='https://rrc.cvc.uab.es/downloads/'
         'Challenge2_Training_Task12_Images.zip',
         save_name='ic13_textdet_train_img.zip',
         md5='a443b9649fda4229c9bc52751bad08fb',
         split=['train'],
         content=['image'],
         mapping=[['ic13_textdet_train_img', 'textdet_imgs/train']]),
     dict(
         url='https://rrc.cvc.uab.es/downloads/'
         'Challenge2_Training_Task1_GT.zip',
         save_name='ic13_textdet_test_img.zip',
         md5='af2e9f070c4c6a1c7bdb7b36bacf23e3',
         split=['test'],
         content=['image'],
         mapping=[['ic13_textdet_test_img', 'textdet_imgs/test']]),
     dict(
         url='https://rrc.cvc.uab.es/downloads/'
         'Challenge2_Test_Task12_Images.zip',
         save_name='ic13_textdet_train_gt.zip',
         md5='f3a425284a66cd67f455d389c972cce4',
         split=['train'],
         content=['annotation'],
         mapping=[['ic13_textdet_train_gt', 'annotations/train']]),
     dict(
         url='https://rrc.cvc.uab.es/downloads/'
         'Challenge2_Test_Task1_GT.zip',
         save_name='ic13_textdet_test_gt.zip',
         md5='3191c34cd6ac28b60f5a7db7030190fb',
         split=['test'],
         content=['annotation'],
         mapping=[['ic13_textdet_test_gt', 'annotations/test']]),
 ])
```

我们首先需要配置数据集获取方法，即 `data_obtainer`，通常来说，内置的 `NaiveDataObtainer` 即可完成绝大部分可以通过直链访问的数据集的下载。`NaiveDataObtainer` 将完成下载、解压、移动文件和重命名等操作。目前，我们暂时不支持自动下载存储在百度或谷歌网盘等需要登陆才能访问资源的数据集。

如下表所示，`data_obtainer` 主要由四个字段构成：

| 字段名     | 含义                                                       |
| ---------- | ---------------------------------------------------------- |
| type       | 数据集获取方法，目前仅支持 `NaiveDataObtainer`             |
| cache_path | 数据集缓存路径，用于存储数据集准备过程中下载的压缩包等文件 |
| data_root  | 数据集存储的根目录                                         |
| files      | 数据集文件列表，用于描述数据集的下载信息                   |

`files` 字段是一个列表，列表中的每个元素都是一个字典，用于描述一个数据集文件的下载信息。如下表所示：

| 字段名            | 含义                                                                 |
| ----------------- | -------------------------------------------------------------------- |
| url               | 数据集文件的下载链接                                                 |
| save_name         | 数据集文件的保存名称                                                 |
| md5 (可选)        | 数据集文件的 md5 值，用于校验下载的文件是否完整                      |
| split （可选）    | 数据集文件所属的数据集划分，如 `train`，`test` 等，该字段可以空缺    |
| content （可选）  | 数据集文件的内容，如 `image`，`annotation` 等，该字段可以空缺        |
| mapping  （可选） | 数据集文件的解压映射，用于指定解压后的文件存储的位置，该字段可以空缺 |

```{note}
为了让 `data_converter.gatherer` 正常运行，我们约定数据集的图片和标注分别储存在 `{taskname}_imgs/{split}/` 和 `annotations/` 下。特别地，对于[多对多的情况](#gatherer)，标注文件需要放置于 `annotations/{split}` 下。
```

##### 配置数据集转换器 `data_converter`

数据集转换器 `data_converter` 主要由标注文件获取器 `gatherer`，原始标注解析器 `parser`，以及数据存储器 `dumper` 组成。其中 `gatherer` 负责将图片与标注文件进行匹配，`parser` 负责将原始标注文件解析为标准格式，`dumper` 负责将标准格式的标注文件存储为 MMOCR 支持的格式。

一般来说，用户无需重新实现新的 `gatherer` 或 `dumper`，但是通常需要根据数据集的标注格式实现新的 `parser`。

我们通过观察获取的 ICDAR2013 数据集文件发现，其每一张图片都有一个对应的 `.txt` 格式的标注文件：

```text
data_root
├── textdet_imgs/train/
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
├── annotations/train/
│   ├── gt_img_1.txt
│   ├── gt_img_2.txt
│   └── ...
```

且每个标注文件名与图片的对应关系为：`gt_img_1.txt` 对应 `img_1.jpg`，以此类推。因此，我们可以使用 `pair_gather` 来进行匹配。

```python
gatherer=dict(
      type='pair_gather',
      suffixes=['.jpg'],
      rule=[r'(\w+)\.jpg', r'gt_\1.txt'])
```

其中，规则 `rule` 是一个[正则表达式对](https://docs.python.org/3/library/re.html)，第一个正则表达式用于匹配图片文件名，第二个正则表达式用于匹配标注文件名。在这里，我们使用 `(\w+)` 来匹配图片文件名，使用 `gt_\1.txt` 来匹配标注文件名，其中 `\1` 表示第一个正则表达式匹配到的内容。即，实现了将 `img_xx.jpg` 替换为 `gt_img_xx.txt` 的功能。

接下来，我们需要实现 `parser`，即将原始标注文件解析为标准格式。通常来说，用户在添加新的数据集前，可以浏览已支持数据集的[详情页](./datasetzoo.md)，并查看是否已有相同格式的数据集。如果已有相同格式的数据集，则可以直接使用该数据集的 `parser`。否则，则需要实现新的格式解析器。

数据格式解析器被统一存储在 `mmocr/datasets/preparers/parsers` 目录下。所有的 `parser` 都需要继承 `BaseParser`，并实现 `parse_file` 或 `parse_files` 方法。

其中，`parse_file()` 方法用于解析单个标注文件，如下代码块所示，`parse_file()` 接受两个参数，`file` 是一个 `Tuple` 类型的变量，包含了由 `gatherer` 获取的图片文件路径和标注文件路径，而 `split` 则会传入当前处理的数据集划分，如 `train` 或 `test`。

```python
def parse_file(self, file: Tuple, split: str) -> Tuple:
    """Convert annotation for a single image.

    Args:
        file (Tuple): A tuple of path to image and annotation
        split (str): Current split.

    Returns:
        Tuple: A tuple of (img_path, instance). Instance is a dict
        containing parsed annotations, which should contain the
        following keys:
        - 'poly' or 'box' (textdet or textspotting)
        - 'text' (textspotting or textrecog)
        - 'ignore' (all task)
    """
```

`parse_file()` 方法的输出是一个 `Tuple` 类型的变量，包含了图片文件路径和标注信息 `instance`。其中，`instance` 是一个字典列表，包含了解析后的标注信息，依据不同的任务类型，该列表中的每一个字典必须包含以下键：

| 键名     | 任务类型               | 说明                                   |
| :------- | :--------------------- | :------------------------------------- |
| box/poly | textdet/textspotting   | 矩形框坐标 `box` 或多边形框坐标 `poly` |
| text     | textrecog/textspotting | 文本内容 `text`                        |
| ignore   | all task               | 是否忽略该样本                         |

以下代码块反映了一个 `parse_file()` 方法返回的数据示例：

```python
('imgs/train/xxx.jpg',
 dict(
    poly=[[[0, 1], [1, 1], [1, 0], [0, 0]]],
    text='hello',
    ignore=False)
)
```

需要注意的是，`parse_file()` 方法解析单个标注文件，并返回单张图片的标注信息，**仅能在标注图像与标注文件满足“多对多”关系时使用**。当仅存在单个标注文件时，可以通过重写 `parse_files()` 方法的方式，直接返回所有样本的数据信息。用户可以参见 `mmocr/datasets/preparers/parsers/totaltext_parser.py` 中的实现。

通过观察 ICDAR2013 数据集的标注文件：

```text
158 128 411 181 "Footpath"
443 128 501 169 "To"
64 200 363 243 "Colchester"
542, 710, 938, 841, "break"
87, 884, 457, 1021, "could"
517, 919, 831, 1024, "save"
```

我们发现内置的 `ICDARTxtTextDetAnnParser` 已经可以满足我们的需求，因此我们可以直接使用该 `parser`，并将其配置到 `preparer` 中。

```python
parser=dict(
     type='ICDARTxtTextDetAnnParser',
     remove_strs=[',', '"'],
     encoding='utf-8',
     format='x1 y1 x2 y2 trans',
     separator=' ',
     mode='xyxy')
```

其中，由于标注文件中混杂了多余的引号 `“”` 和逗号 `,`，我们通过指定 `remove_strs=[',', '"']` 来进行移除。另外，我们在 `format` 中指定了标注文件的格式，其中 `x1 y1 x2 y2 trans` 表示标注文件中的每一行包含了四个坐标和一个文本内容，且坐标和文本内容之间使用空格分隔（`separator`=' '）。另外，我们需要指定 `mode` 为 `xyxy`，表示标注文件中的坐标是左上角和右下角的坐标，这样以来，`ICDARTxtTextDetAnnParser` 即可将该格式的标注解析为 MMOCR 统一格式。

最后，数据格式转换器 `data_converter` 的完整配置如下：

```python
data_converter = dict(
  type='TextDetDataConverter',
  splits=['train', 'test'],
  data_root=data_root,
  gatherer=dict(
      type='pair_gather',
      suffixes=['.jpg'],
      rule=[r'(\w+)\.jpg', r'gt_\1.txt']),
  parser=dict(
     type='ICDARTxtTextDetAnnParser',
     remove_strs=[',', '"'],
     encoding='utf-8',
     format='x1 y1 x2 y2 trans',
     separator=' ',
     mode='xyxy'),
  dumper=dict(type='JsonDumper'))
```

##### 配置基础配置生成器 `config_generator`

为了在数据集准备完毕后可以自动生成基础配置，我们还需要配置一下对应任务的 `config_generator`。目前，MMOCR 按任务实现了 `TextDetConfigGenerator`、`TextRecogConfigGenerator` 和 `TextSpottingConfigGenerator`。它们支持的主要参数如下：

| 字段名      | 含义                                                                                                                                                |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| data_root   | 数据集存储的根目录                                                                                                                                  |
| train_anns  | 配置文件内训练集标注的路径。若不指定，则默认为 `[dict(ann_file='{taskname}_train.json', dataset_postfix='']`。                                      |
| val_anns    | 配置文件内验证集标注的路径。若不指定，则默认为空。                                                                                                  |
| test_anns   | 配置文件内测试集标注的路径。若不指定，则默认指向 `[dict(ann_file='{taskname}_test.json', dataset_postfix='']`。                                     |
| config_path | 算法库存放配置文件的路径，配置生成器会将默认配置写入 `{config_path}/{taskname}/_base_/datasets/{dataset_name}.py` 下。若不指定，则默认为 `configs/` |

在本例中，我们只需要设置一下 `TextDetConfigGenerator` 的 `data_root` 字段，其它字段保持默认即可。

```python
config_generator = dict(type='TextDetConfigGenerator', data_root=data_root)
```

假如数据集比较特殊，标注存在着几个变体，配置生成器也支持在基础配置中生成指向各自变体的变量，但这需要用户在设置时用不同的 `dataset_postfix` 区分。例如，ICDAR 2015 文字识别数据的测试集就存在着原版和 1811 两种标注版本，我们可以在 `test_anns` 中指定它们，如下所示：

```python
config_generator = dict(
    type='TextRecogConfigGenerator',
    data_root=data_root,
    test_anns=[
        dict(ann_file='textrecog_test.json'),
        dict(dataset_postfix='857', ann_file='textrecog_test_857.json')
    ])
```

配置生成器会生成以下配置：

```python
icdar2015_textrecog_data_root = 'data/icdar2015'

icdar2015_textrecog_train = dict(
    type='OCRDataset',
    data_root=icdar2015_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

icdar2015_textrecog_test = dict(
    type='OCRDataset',
    data_root=icdar2015_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)

icdar2015_1811_textrecog_test = dict(
    type='OCRDataset',
    data_root=icdar2015_textrecog_data_root,
    ann_file='textrecog_test_1811.json',
    test_mode=True,
    pipeline=None)
```

#### 添加标注示例

最后，我们可以在 `dataset_zoo/icdar2013/` 目录下添加标注示例文件 `sample_anno.md` 以帮助文档脚本在生成文档时添加标注示例，标注示例文件是一个 Markdown 文件，其内容通常包含了单个样本的原始数据格式。例如，以下代码块展示了 ICDAR2013 数据集的数据样例文件：

````markdown
  **Text Detection**

  ```text
  # train split
  # x1 y1 x2 y2 "transcript"

  158 128 411 181 "Footpath"
  443 128 501 169 "To"
  64 200 363 243 "Colchester"

  # test split
  # x1, y1, x2, y2, "transcript"

  38, 43, 920, 215, "Tiredness"
  275, 264, 665, 450, "kills"
  0, 699, 77, 830, "A"
  ```
````

### 添加私有数据集

待更新...
