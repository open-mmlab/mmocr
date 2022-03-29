
# Text Detection

## Overview

The structure of the text detection dataset directory is organized as follows.

```text
├── ctw1500
│   ├── annotations
│   ├── imgs
│   ├── instances_test.json
│   └── instances_training.json
├── icdar2015
│   ├── imgs
│   ├── instances_test.json
│   └── instances_training.json
├── icdar2017
│   ├── imgs
│   ├── instances_training.json
│   └── instances_val.json
├── synthtext
│   ├── imgs
│   └── instances_training.lmdb
│       ├── data.mdb
│       └── lock.mdb
├── textocr
│   ├── train
│   ├── instances_training.json
│   └── instances_val.json
├── totaltext
│   ├── imgs
│   ├── instances_test.json
│   └── instances_training.json
├── CurvedSynText150k
│   ├── syntext_word_eng
│   ├── emcs_imgs
│   └── instances_training.json
|── funsd
|   ├── annotations
│   ├── imgs
│   ├── instances_test.json
│   └── instances_training.json
```

|      Dataset      |                                                                                                                                     Images                                                                                                                                     |                                                                                                                                                                                                                              |                                       Annotation Files                                       |                                                                                                |       |
| :---------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: | :---: |
|                   |                                                                                                                                                                                                                                                                                |                                                                                                           training                                                                                                           |                                          validation                                          |                                            testing                                             |       |
|      CTW1500      |                                                                                                         [homepage](https://github.com/Yuliang-Liu/Curve-Text-Detector)                                                                                                         |                                                                                                              -                                                                                                               |                                              -                                               |                                               -                                                |
|     ICDAR2015     |                                                                                                             [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)                                                                                                             |                                                            [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json)                                                            |                                              -                                               | [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json) |
|     ICDAR2017     |                                                                                                             [homepage](https://rrc.cvc.uab.es/?ch=8&com=downloads)                                                                                                             |                                                            [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_training.json)                                                            | [instances_val.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_val.json) |                                               -                                                |       |  |
|     Synthtext     |                                                                                                          [homepage](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)                                                                                                          | instances_training.lmdb ([data.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/data.mdb), [lock.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/lock.mdb)) |                                              -                                               |                                               -                                                |
|      TextOCR      |                                                                                                                [homepage](https://textvqa.org/textocr/dataset)                                                                                                                 |                                                                                                              -                                                                                                               |                                              -                                               |                                               -                                                |
|     Totaltext     |                                                                                                           [homepage](https://github.com/cs-chan/Total-Text-Dataset)                                                                                                            |                                                                                                              -                                                                                                               |                                              -                                               |                                               -                                                |
| CurvedSynText150k | [homepage](https://github.com/aim-uofa/AdelaiDet/blob/master/datasets/README.md) \| [Part1](https://drive.google.com/file/d/1OSJ-zId2h3t_-I7g_wUkrK-VqQy153Kj/view?usp=sharing) \| [Part2](https://drive.google.com/file/d/1EzkcOlIgEp5wmEubvHb7-J5EImHExYgY/view?usp=sharing) |                                                          [instances_training.json](https://download.openmmlab.com/mmocr/data/curvedsyntext/instances_training.json)                                                          |                                              -                                               |                                               -                                                |
|       FUNSD       |                                                                                                              [homepage](https://guillaumejaume.github.io/FUNSD/)                                                                                                               |                                                                                                              -                                                                                                               |                                              -                                               |                                               -                                                |
|      VinText      |                                                                                                            [homepage](https://github.com/VinAIResearch/dict-guided)                                                                                                            |                                                                                                              -                                                                                                               |                                              -                                               |                                               -                                                |



## Important Note

:::{note}
**For users who want to train models on CTW1500, ICDAR 2015/2017, and Totaltext dataset,** there might be some images containing orientation info in EXIF data. The default OpenCV
backend used in MMCV would read them and apply the rotation on the images.  However, their gold annotations are made on the raw pixels, and such
inconsistency results in false examples in the training set. Therefore, users should use `dict(type='LoadImageFromFile', color_type='color_ignore_orientation')` in pipelines to change MMCV's default loading behaviour. (see [DBNet's pipeline config](https://github.com/open-mmlab/mmocr/blob/main/configs/_base_/det_pipelines/dbnet_pipeline.py) for example)
:::

## Preparation Steps
### ICDAR 2015
- Step0: Read [Important Note](#important-note)
- Step1: Download `ch4_training_images.zip`, `ch4_test_images.zip`, `ch4_training_localization_transcription_gt.zip`, `Challenge4_Test_Task1_GT.zip` from [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)
- Step2:
```bash
mkdir icdar2015 && cd icdar2015
mkdir imgs && mkdir annotations
# For images,
mv ch4_training_images imgs/training
mv ch4_test_images imgs/test
# For annotations,
mv ch4_training_localization_transcription_gt annotations/training
mv Challenge4_Test_Task1_GT annotations/test
```
- Step3: Download [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json) and [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json) and move them to `icdar2015`
- Or, generate `instances_training.json` and `instances_test.json` with following command:
```bash
python tools/data/textdet/icdar_converter.py /path/to/icdar2015 -o /path/to/icdar2015 -d icdar2015 --split-list training test
```

### ICDAR 2017
- Follow similar steps as [ICDAR 2015](#icdar-2015).

### CTW1500
- Step0: Read [Important Note](#important-note)
- Step1: Download `train_images.zip`, `test_images.zip`, `train_labels.zip`, `test_labels.zip` from [github](https://github.com/Yuliang-Liu/Curve-Text-Detector)
```bash
mkdir ctw1500 && cd ctw1500
mkdir imgs && mkdir annotations

# For annotations
cd annotations
wget -O train_labels.zip https://universityofadelaide.box.com/shared/static/jikuazluzyj4lq6umzei7m2ppmt3afyw.zip
wget -O test_labels.zip https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/download
unzip train_labels.zip && mv ctw1500_train_labels training
unzip test_labels.zip -d test
cd ..
# For images
cd imgs
wget -O train_images.zip https://universityofadelaide.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip
wget -O test_images.zip https://universityofadelaide.box.com/shared/static/t4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip
unzip train_images.zip && mv train_images training
unzip test_images.zip && mv test_images test
```
- Step2: Generate `instances_training.json` and `instances_test.json` with following command:

```bash
python tools/data/textdet/ctw1500_converter.py /path/to/ctw1500 -o /path/to/ctw1500 --split-list training test
```

### SynthText

- Download [data.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/data.mdb) and [lock.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/lock.mdb) to `synthtext/instances_training.lmdb/`.

### TextOCR
- Step1: Download [train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), [TextOCR_0.1_train.json](https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json) and [TextOCR_0.1_val.json](https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json) to `textocr/`.
```bash
mkdir textocr && cd textocr

# Download TextOCR dataset
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json
wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json

# For images
unzip -q train_val_images.zip
mv train_images train
```
- Step2: Generate `instances_training.json` and `instances_val.json` with the following command:
```bash
python tools/data/textdet/textocr_converter.py /path/to/textocr
```
### Totaltext
- Step0: Read [Important Note](#important-note)
- Step1: Download `totaltext.zip` from [github dataset](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) and `groundtruth_text.zip` from [github Groundtruth](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Groundtruth/Text) (Our totaltext_converter.py supports groundtruth with both .mat and .txt format).
```bash
mkdir totaltext && cd totaltext
mkdir imgs && mkdir annotations

# For images
# in ./totaltext
unzip totaltext.zip
mv Images/Train imgs/training
mv Images/Test imgs/test

# For annotations
unzip groundtruth_text.zip
cd Groundtruth
mv Polygon/Train ../annotations/training
mv Polygon/Test ../annotations/test

```
- Step2: Generate `instances_training.json` and `instances_test.json` with the following command:
```bash
python tools/data/textdet/totaltext_converter.py /path/to/totaltext -o /path/to/totaltext --split-list training test
```

### CurvedSynText150k

- Step1: Download [syntext1.zip](https://drive.google.com/file/d/1OSJ-zId2h3t_-I7g_wUkrK-VqQy153Kj/view?usp=sharing) and [syntext2.zip](https://drive.google.com/file/d/1EzkcOlIgEp5wmEubvHb7-J5EImHExYgY/view?usp=sharing) to `CurvedSynText150k/`.
- Step2:

```bash
unzip -q syntext1.zip
mv train.json train1.json
unzip images.zip
rm images.zip

unzip -q syntext2.zip
mv train.json train2.json
unzip images.zip
rm images.zip
```

- Step3: Download [instances_training.json](https://download.openmmlab.com/mmocr/data/curvedsyntext/instances_training.json) to `CurvedSynText150k/`
- Or, generate `instances_training.json` with following command:

```bash
python tools/data/common/curvedsyntext_converter.py PATH/TO/CurvedSynText150k --nproc 4
```

### FUNSD

- Step1: Download [dataset.zip](https://guillaumejaume.github.io/FUNSD/dataset.zip) to `funsd/`.

```bash
mkdir funsd && cd funsd

# Download FUNSD dataset
wget https://guillaumejaume.github.io/FUNSD/dataset.zip
unzip -q dataset.zip

# For images
mv dataset/training_data/images imgs && mv dataset/testing_data/images/* imgs/

# For annotations
mkdir annotations
mv dataset/training_data/annotations annotations/training && mv dataset/testing_data/annotations annotations/test

rm dataset.zip && rm -rf dataset
```

- Step2: Generate `instances_training.json` and `instances_test.json` with following command:

```bash
python tools/data/textdet/funsd_converter.py PATH/TO/funsd --nproc 4
```

### VinText
- Step1: Download [vintext.zip](https://drive.google.com/drive/my-drive) to `vintext`
```bash
mkdir vintext && cd vintext
# Download dataset from google drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml" -O vintext.zip && rm -rf /tmp/cookies.txt
# Extract images and annotations
unzip -q vintext.zip && rm vintext.zip
mv vietnamese/labels ./ && mv vietnamese/test_image ./ && mv vietnamese/train_images ./ && mv vietnamese/unseen_test_images ./
rm -rf vietnamese
# Rename files
mv labels annotations && mv test_image test && mv train_images  training && mv unseen_test_images  unseen_test
mkdir imgs
mv training imgs/ && mv test imgs/ && mv unseen_test imgs/
```
- Step2: Generate `instances_training.json`, `instances_test.json` and `instances_unseen_test.json`
```bash
python tools/data/textdet/vintext_converter.py PATH/TO/vintext --nproc 4
```
After running the above codes, the directory structure should be as follows:
```text
|── vintext
|   ├── annotations
│   ├── imgs
│   ├── instances_test.json
│   ├── instances_unseen_test.json
│   └── instances_training.json
```