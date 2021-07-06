# Datasets Preparation

This page lists the datasets which are commonly used in text detection, text recognition and key information extraction, and their download links.

<!-- TOC -->

- [Datasets Preparation](#datasets-preparation)
  - [Text Detection](#text-detection)
  - [Text Recognition](#text-recognition)
  - [Key Information Extraction](#key-information-extraction)
  - [Named Entity Recognition](#named-entity-recognition)
    - [CLUENER2020](#cluener2020)

<!-- /TOC -->

## Text Detection

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
├── textocr
│   ├── train
│   ├── instances_training.json
│   └── instances_val.json
├── totaltext
│   ├── imgs
│   ├── instances_test.json
│   └── instances_training.json
```

|  Dataset  |                             Images                             |                                                                                      |                                                                                                        |            Annotation Files             |                                                                                                |
| :-------: | :------------------------------------------------------------: | :----------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :-------------------------------------: | :--------------------------------------------------------------------------------------------: |
|      |                                                                |                                                                                      |                                                training                                                |               validation                |                                            testing                                             |       |
|  CTW1500  | [homepage](https://github.com/Yuliang-Liu/Curve-Text-Detector) |                                        |                    -                    |                    -                    |                    -                    |
| ICDAR2015 | [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)     |                                                                                      | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json) |                    -                    | [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json) |
| ICDAR2017 | [homepage](https://rrc.cvc.uab.es/?ch=8&com=downloads)     | [renamed_imgs](https://download.openmmlab.com/mmocr/data/icdar2017/renamed_imgs.tar) | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_training.json) | [instances_val.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_val.json) | - |       |       |
| Synthtext | [homepage](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)  |                                                                                      | [instances_training.lmdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb) |                    -                    |
| TextOCR | [homepage](https://textvqa.org/textocr/dataset)  |                                                                                      | - |                    -                    | -
| Totaltext | [homepage](https://github.com/cs-chan/Total-Text-Dataset)  |                                                                                      | - |                    -                    | -

- For `icdar2015`:
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

- For `icdar2017`:
  - To avoid the effect of rotation when load `jpg` with opencv, We provide re-saved `png` format image in [renamed_images](https://download.openmmlab.com/mmocr/data/icdar2017/renamed_imgs.tar). You can copy these images to `imgs`.

- For `ctw1500`:
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
- For `TextOCR`:
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
- For `Totaltext`:
  - Step1: Download `totaltext.zip` from [github dataset](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) and `groundtruth_text.zip` from [github Groundtruth](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Groundtruth/Text) (We recommend downloading the text groundtruth with .mat format since our totaltext_converter.py supports groundtruth with .mat format only).
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
## Text Recognition

**The structure of the text recognition dataset directory is organized as follows.**

```text
├── mixture
│   ├── coco_text
│   │   ├── train_label.txt
│   │   ├── train_words
│   ├── icdar_2011
│   │   ├── training_label.txt
│   │   ├── Challenge1_Training_Task3_Images_GT
│   ├── icdar_2013
│   │   ├── train_label.txt
│   │   ├── test_label_1015.txt
│   │   ├── test_label_1095.txt
│   │   ├── Challenge2_Training_Task3_Images_GT
│   │   ├── Challenge2_Test_Task3_Images
│   ├── icdar_2015
│   │   ├── train_label.txt
│   │   ├── test_label.txt
│   │   ├── ch4_training_word_images_gt
│   │   ├── ch4_test_word_images_gt
│   ├── III5K
│   │   ├── train_label.txt
│   │   ├── test_label.txt
│   │   ├── train
│   │   ├── test
│   ├── ct80
│   │   ├── test_label.txt
│   │   ├── image
│   ├── svt
│   │   ├── test_label.txt
│   │   ├── image
│   ├── svtp
│   │   ├── test_label.txt
│   │   ├── image
│   ├── Syn90k
│   │   ├── shuffle_labels.txt
│   │   ├── label.txt
│   │   ├── label.lmdb
│   │   ├── mnt
│   ├── SynthText
│   │   ├── shuffle_labels.txt
│   │   ├── instances_train.txt
│   │   ├── label.txt
│   │   ├── label.lmdb
│   │   ├── synthtext
│   ├── SynthAdd
│   │   ├── label.txt
│   │   ├── label.lmdb
│   │   ├── SynthText_Add
│   ├── TextOCR
│   │   ├── image
│   │   ├── train_label.txt
│   │   ├── val_label.txt
│   ├── Totaltext
│   │   ├── imgs
│   │   ├── annotations
│   │   ├── train_label.txt
│   │   ├── test_label.txt
```

|  Dataset   |                                        images                                         |                                                                                                                                            annotation file                                                                                                                                             |                                             annotation file                                             |
| :--------: | :-----------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: |
|       |                                                                                       |                                                                                                                                                training                                                                                                                                                |                                                  test                                                   |
| coco_text  |                [homepage](https://rrc.cvc.uab.es/?ch=5&com=downloads)                 |                                                                                                     [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_label.txt)                                                                                                     |                                                    -                                                    |       |
| icdar_2011 | [homepage](http://www.cvc.uab.es/icdar2011competition/?com=downloads)         |                                                                                                    [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt)                                                                                                     |                                                    -                                                    |       |
| icdar_2013 |              [homepage](https://rrc.cvc.uab.es/?ch=2&com=downloads)                 |                                                                                                    [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/train_label.txt)                                                                                                     | [test_label_1015.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/test_label_1015.txt) |       |
| icdar_2015 |               [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)                 |                                                                                                    [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt)                                                                                                     |      [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt)      |       |
|   IIIT5K   |    [homepage](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)     |                                                                                                      [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/train_label.txt)                                                                                                       |        [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)        |       |
|    ct80    |                                            -                                           |                                                                                                                                                   -                                                                                                                                                    |         [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/ct80/test_label.txt)         |       |
|    svt     |[homepage](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) |                                                                                                                                                   -                                                                                                                                                    |         [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svt/test_label.txt)          |       |
|    svtp    |                              -                                           |                                                                                                                                                   -                                                                                                                                                    |         [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svtp/test_label.txt)         |       |
|  Syn90k  |               [homepage](https://www.robots.ox.ac.uk/~vgg/data/text/)                |                                                       [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/shuffle_labels.txt) \| [label.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/label.txt)                                                       |                                                    -                                                    |       |
| SynthText  |           [homepage](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)              | [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/shuffle_labels.txt) \| [instances_train.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/instances_train.txt) \| [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/label.txt) |                                                    -                                                    |       |
|  SynthAdd  |  [SynthText_Add.zip](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg)  (code:627x)   |                                                                                                           [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/label.txt)                                                                                                            |                                                    -                                                    |       |
|  TextOCR  |  [homepage](https://textvqa.org/textocr/dataset)   |                                                                                                           -                                                                                                           |                                                    -                                                    |       |
|  Totaltext  |  [homepage](https://github.com/cs-chan/Total-Text-Dataset)   |                                                                                                           -                                                                                                           |                                                    -                                                    |       |

- For `icdar_2013`:
  - Step1: Download `Challenge2_Test_Task3_Images.zip` and `Challenge2_Training_Task3_Images_GT.zip` from [homepage](https://rrc.cvc.uab.es/?ch=2&com=downloads)
  - Step2: Download [test_label_1015.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/test_label_1015.txt) and [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/train_label.txt)
- For `icdar_2015`:
  - Step1: Download `ch4_training_word_images_gt.zip` and `ch4_test_word_images_gt.zip` from [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)
  - Step2: Download [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt) and [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt)
- For `IIIT5K`:
  - Step1: Download `IIIT5K-Word_V3.0.tar.gz` from [homepage](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)
  - Step2: Download [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/train_label.txt) and [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)
- For `svt`:
  - Step1: Download `svt.zip` form [homepage](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
  - Step2: Download [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svt/test_label.txt)
  - Step3:
  ```bash
  python tools/data/textrecog/svt_converter.py <download_svt_dir_path>
  ```
- For `ct80`:
  - Step1: Download [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/ct80/test_label.txt)
- For `svtp`:
  - Step1: Download [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svtp/test_label.txt)
- For `coco_text`:
  - Step1: Download from [homepage](https://rrc.cvc.uab.es/?ch=5&com=downloads)
  - Step2: Download [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_label.txt)
- For `Syn90k`:
  - Step1: Download `mjsynth.tar.gz` from [homepage](https://www.robots.ox.ac.uk/~vgg/data/text/)
  - Step2: Download [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/shuffle_labels.txt)
  - Step3:

  ```bash
  mkdir Syn90k && cd Syn90k

  mv /path/to/mjsynth.tar.gz .

  tar -xzf mjsynth.tar.gz

  mv /path/to/shuffle_labels.txt .

  # create soft link
  cd /path/to/mmocr/data/mixture

  ln -s /path/to/Syn90k Syn90k
  ```

- For `SynthText`:
  - Step1: Download `SynthText.zip` from [homepage](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
  - Step2:

  ```bash
  mkdir SynthText && cd SynthText
  mv /path/to/SynthText.zip SynthText
  unzip SynthText.zip
  mv SynthText synthtext

  mv /path/to/shuffle_labels.txt .

  # create soft link
  cd /path/to/mmocr/data/mixture
  ln -s /path/to/SynthText SynthText
  ```
  - Step3:
  Generate cropped images and labels:

  ```bash
  cd /path/to/mmocr

  python tools/data/textrecog/synthtext_converter.py data/mixture/SynthText/gt.mat data/mixture/SynthText/ data/mixture/SynthText/synthtext/SynthText_patch_horizontal --n_proc 8
  ```

- For `SynthAdd`:
  - Step1: Download `SynthText_Add.zip` from [SynthAdd](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg) (code:627x))
  - Step2: Download [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/label.txt)
  - Step3:

  ```bash
  mkdir SynthAdd && cd SynthAdd

  mv /path/to/SynthText_Add.zip .

  unzip SynthText_Add.zip

  mv /path/to/label.txt .

  # create soft link
  cd /path/to/mmocr/data/mixture

  ln -s /path/to/SynthAdd SynthAdd
  ```
  **Note:**
To convert label file with `txt` format to `lmdb` format,
```bash
python tools/data/utils/txt2lmdb.py -i <txt_label_path> -o <lmdb_label_path>
```
For example,
```bash
python tools/data/utils/txt2lmdb.py -i data/mixture/Syn90k/label.txt -o data/mixture/Syn90k/label.lmdb
```
- For `TextOCR`:
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
  - Step2: Generate `train_label.txt`, `val_label.txt` and crop images using 4 processes with the following command:
  ```bash
  python tools/data/textrecog/textocr_converter.py /path/to/textocr 4
  ```


- For `Totaltext`:
  - Step1: Download `totaltext.zip` from [github dataset](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) and `groundtruth_text.zip` from [github Groundtruth](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Groundtruth/Text) (We recommend downloading the text groundtruth with .mat format since our totaltext_converter.py supports groundtruth with .mat format only).
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
  - Step2: Generate cropped images, `train_label.txt` and `test_label.txt` with the following command (the cropped images will be saved to `data/totaltext/dst_imgs/`.):
  ```bash
  python tools/data/textrecog/totaltext_converter.py /path/to/totaltext -o /path/to/totaltext --split-list training test
  ```



## Key Information Extraction

The structure of the key information extraction dataset directory is organized as follows.

```text
└── wildreceipt
  ├── class_list.txt
  ├── dict.txt
  ├── image_files
  ├── test.txt
  └── train.txt
```

- Download [wildreceipt.tar](https://download.openmmlab.com/mmocr/data/wildreceipt.tar)


## Named Entity Recognition

### CLUENER2020

The structure of the named entity recognition dataset directory is organized as follows.

```text
└── cluener2020
  ├── cluener_predict.json
  ├── dev.json
  ├── README.md
  ├── test.json
  ├── train.json
  └── vocab.txt

```
- Download [cluener_public.zip](https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip)

- Download [vocab.txt](https://download.openmmlab.com/mmocr/data/cluener2020/vocab.txt) and move `vocab.txt` to `cluener2020`
