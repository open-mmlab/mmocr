# Dataset Preparer (Beta)

```{note}
Dataset Preparer is still in beta version and might not be stable enough. You
are welcome to try it out and report any issues to us.
```

## One-click data preparation script

MMOCR provides a unified one-stop data preparation script `prepare_dataset.py`.

Only one line of command is needed to complete the data download, decompression, and format conversion.

```bash
python tools/dataset_converters/prepare_dataset.py [$DATASET_NAME] --task [$TASK] --nproc [$NPROC]
```

| ARGS         | Type | Description                                                                                                                               |
| ------------ | ---- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| dataset_name | str  | (required) dataset name.                                                                                                                  |
| --task       | str  | Convert the dataset to the format of a specified task supported by MMOCR. options are: 'textdet', 'textrecog', 'textspotting', and 'kie'. |
| --nproc      | int  | Number of processes to be used. Defaults to 4.                                                                                            |

For example, the following command shows how to use the script to prepare the ICDAR2015 dataset for text detection task.

```bash
python tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet
```

Also, the script supports preparing multiple datasets at the same time. For example, the following command shows how to prepare the ICDAR2015 and TotalText datasets for text recognition task.

```bash
python tools/dataset_converters/prepare_dataset.py icdar2015 totaltext --task textrecog
```

To check the supported datasets in MMOCR, please refer to [Dataset Zoo](./datasetzoo.md).

## Advanced Usage

### Configuration of Dataset Preparer

Dataset preparer uses a modular design to enhance extensibility, which allows users to extend it to other public or private datasets easily. The configuration files of the dataset preparers are stored in the `dataset_zoo/`, where all the configs of currently supported datasets can be found here. The directory structure is as follows:

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

`metafile.yml` is the metafile of the dataset, which contains the basic information of the dataset, including the year of publication, the author of the paper, and other information such as license. The other files named by the task are the configuration files of the dataset preparer, which are used to configure the download, decompression, format conversion, etc. of the dataset. These configs are in Python format, and their usage is completely consistent with the configuration files in MMOCR repo. See [Configuration File Documentation](../config.md) for detailed usage.

#### Metafile

Take the ICDAR2015 dataset as an example, the `metafile.yml` stores the basic information of the dataset:

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

It is not mandatory to use the metafile in the dataset preparation process (so users can ignore this file when preparing private datasets), but in order to better understand the information of each public dataset, we recommend that users read the metafile before preparing the dataset, which will help to understand whether the datasets meet their needs.

#### Config of Dataset Preparer

Next, we will introduce the conventional fields and usage of the dataset preparer configuration files.

In the configuration files, there are two fields `data_root` and `cache_path`, which are used to store the converted dataset and the temporary files such as the archived files downloaded during the data preparation process.

```python
data_root = './data/icdar2015'
cache_path = './data/cache'
```

Data preparation usually contains two steps: "raw data preparation" and "format conversion and saving". Therefore, we use the `data_obtainer` and `data_converter` to configure the behavior of these two steps. In some cases, users can also ignore `data_converter` to only download and decompress the raw data, without performing format conversion and saving. Or, for the local stored dataset, use ignore `data_obtainer` to only perform format conversion and saving.

Take the text detection task of the ICDAR2015 dataset (`dataset_zoo/icdar2015/textdet.py`) as an example:

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

The default type of `data_obtainer` is `NaiveDataObtainer`, which mainly downloads and decompresses the original files to the specified directory. Here, we configure the URL, save name, MD5 value, etc. of the original dataset files through the `files` parameter. The `mapping` parameter is used to specify the path where the data is decompressed or moved. In addition, the two optional parameters `split` and `content` respectively indicate the content type stored in the compressed file and the corresponding dataset.

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

`data_converter` is responsible for loading and converting the original to the format supported by MMOCR. We provide a number of built-in data converters for different tasks, such as `TextDetDataConverter`, `TextRecogDataConverter`, `TextSpottingDataConverter`, and `WildReceiptConverter` (Since we only support WildReceipt dataset for KIE task at present, we only provide this converter for now).

Take the text detection task as an example, `TextDetDataConverter` mainly completes the following work:

- Collect and match the images and original annotation files, such as the image `img_1.jpg` and the annotation `gt_img_1.txt`
- Load and parse the original annotations to obtain necessary information such as the bounding box and text
- Convert the parsed data to the format supported by MMOCR
- Dump the converted data to the specified path and format

The above steps can be configured separately through `gatherer`, `parser`, `dumper`.

Specifically, the `gatherer` is used to collect and match the images and annotations in the original dataset. Typically, there are two relations between images and annotations, one is many-to-many, the other is many-to-one.

```text
many-to-many
├── img_1.jpg
├── gt_img_1.txt
├── img_2.jpg
├── gt_img_2.txt
├── img_3.JPG
├── gt_img_3.txt

one-to-many
├── img_1.jpg
├── img_2.jpg
├── img_3.JPG
├── gt.txt
```

Therefore, we provide two built-in gatherers, `pair_gather` and `mono_gather`, to handle the two cases. `pair_gather` is used for the case of many-to-many, and `mono_gather` is used for the case of one-to-many. `pair_gather` needs to specify the `suffixes` parameter to indicate the suffix of the image, such as `suffixes=[.jpg,.JPG]` in the above example. In addition, we need to specify the corresponding relationship between the image and the annotation file through the regular expression, such as `rule=[r'img_(\d+)\.([jJ][pP][gG])'，r'gt_img_\1.txt']` in the above example. Where `\d+` is used to match the serial number of the image, `([jJ][pP][gG])` is used to match the suffix of the image, and `\_1` matches the serial number of the image and the serial number of the annotation file.

When the image and annotation file are matched, the original annotations will be parsed. Since the annotation format is usually varied from dataset to dataset, the parsers are usually dataset related. Then, the parser will pack the required data into the MMOCR format.

Finally, we can specify the dumpers to decide the data format. Currently, we only support `JsonDumper` and `WildreceiptOpensetDumper`, where the former is used to save the data in the standard MMOCR Json format, and the latter is used to save the data in the Wildreceipt format. In the future, we plan to support `LMDBDumper` to save the annotation files in LMDB format.

### Use DataPreparer to prepare customized dataset

\[Coming Soon\]
