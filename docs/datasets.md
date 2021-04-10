# Datasets Preparation

This page lists the datasets which are commonly used in text detection, text recognition and key information extraction, and their download links.

<!-- TOC -->

- [Datasets Preparation](#datasets-preparation)
  - [Text Detection](#text-detection)
  - [Text Recognition](#text-recognition)
  - [Key Information Extraction](#key-information-extraction)

<!-- /TOC -->

## Text Detection

The structure of the text detection dataset directory is organized as follows.

```text
├── ctw1500
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
```

|  Dataset  |                             Images                             |                                                                                      |                                                                                                        |            Annotation Files             |                                                                                                |
| :-------: | :------------------------------------------------------------: | :----------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :-------------------------------------: | :--------------------------------------------------------------------------------------------: |
|      |                                                                |                                                                                      |                                                training                                                |               validation                |                                            testing                                             |       |
|  CTW1500  | [homepage](https://github.com/Yuliang-Liu/Curve-Text-Detector) |                                                                                      |  [instances_training.json](https://download.openmmlab.com/mmocr/data/ctw1500/instances_training.json)  |                    -                    |  [instances_test.json](https://download.openmmlab.com/mmocr/data/ctw1500/instances_test.json)  |
| ICDAR2015 | [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)     |                                                                                      | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json) |                    -                    | [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json) |
| ICDAR2017 | [homepage](https://rrc.cvc.uab.es/?ch=8&com=downloads)     | [renamed_imgs](https://download.openmmlab.com/mmocr/data/icdar2017/renamed_imgs.tar) | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_training.json) | [instances_val.json](https://openmmlab) | [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_test.json) |       |       |
| Synthtext | [homepage](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)  |                                                                                      | [instances_training.lmdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb) |                    -                    |

- For `icdar2015`:
  - Step1: Download `ch4_training_images.zip` and `ch4_test_images.zip` from [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)
  - Step2: Download [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json) and [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json)
  - Step3:

  ```bash
  mkdir icdar2015 && cd icdar2015
  mv /path/to/instances_training.json .
  mv /path/to/instances_test.json .

  mkdir imgs && cd imgs
  ln -s /path/to/ch4_training_images training
  ln -s /path/to/ch4_test_images test
  ```

- For `icdar2017`:
  - To avoid the effect of rotation when load `jpg` with opencv, We provide re-saved `png` format image in [renamed_images](https://download.openmmlab.com/mmocr/data/icdar2017/renamed_imgs.tar). You can copy these images to `imgs`.

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
│   │   ├── label.lmdb
│   │   ├── mnt
│   ├── SynthText
│   │   ├── shuffle_labels.txt
│   │   ├── instances_train.txt
│   │   ├── label.lmdb
│   │   ├── synthtext
│   ├── SynthAdd
│   │   ├── label.txt
│   │   ├── SynthText_Add
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
|  Syn90k  |               [homepage](https://www.robots.ox.ac.uk/~vgg/data/text/)                |                                                       [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/shuffle_labels.txt) \| [label.lmdb](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/label.lmdb)                                                       |                                                    -                                                    |       |
| SynthText  |           [homepage](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)              | [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/shuffle_labels.txt) \| [instances_train.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/instances_train.txt) \| [label.lmdb](https://download.openmmlab.com/mmocr/data/mixture/SynthText/label.lmdb) |                                                    -                                                    |       |
|  SynthAdd  |  [SynthText_Add.zip](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg)  (code:627x)   |                                                                                                           [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/label.txt)                                                                                                            |                                                    -                                                    |       |

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
  - Step2: Download [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/shuffle_labels.txt)
  - Step3: Download [instances_train.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/instances_train.txt)
  - Step4:

  ```bash
  unzip SynthText.zip

  cd SynthText

  mv /path/to/shuffle_labels.txt .

  # create soft link
  cd /path/to/mmocr/data/mixture

  ln -s /path/to/SynthText SynthText
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

## Key Information Extraction

The structure of the key information extraction dataset directory is organized as follows.

```text
└── wildreceipt
  ├── anno_files
  ├── class_list.txt
  ├── dict.txt
  ├── image_files
  ├── test.txt
  └── train.txt
```

- Download [wildreceipt.tar](https://download.openmmlab.com/mmocr/data/wildreceipt.tar)
