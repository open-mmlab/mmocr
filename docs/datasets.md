# Datasets Preparation
This page lists the datasets which are commonly used in text detection, text recognition and key information extraction, and their download links.

## Text Detection
**The structure of the text detection dataset directory is organized as follows.**
```
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
│   ├── instances_training.json
│   ├── instances_training.txt
│   └── instances_training.lmdb
```
|  Dataset  |   |           Images           |   |                                              |             Annotation Files             |                                          |   | Note |   |
|:---------:|:-:|:--------------------------:|:-:|:--------------------------------------------:|:---------------------------------------:|:----------------------------------------:|:-:|:----:|---|
|           |   |                            |   | training                                     | validation                               | testing                                  |   |      |   |
| CTW1500   |   | [link](https://github.com/Yuliang-Liu/Curve-Text-Detector) |   | [instances_training.json](https://download.openmmlab.com/mmocr/data/ctw1500/instances_training.json) | -                                       | [instances_test.json](https://download.openmmlab.com/mmocr/data/ctw1500/instances_test.json) |   |      |   |
| ICDAR2015 |   | [link](https://rrc.cvc.uab.es/?ch=4&com=downloads) |   | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json) |                    -                     | [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json) |   |      |   |
| ICDAR2017 |   | [link](https://rrc.cvc.uab.es/?ch=8&com=downloads) |   | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_training.json) | [instances_val.json](https://openmmlab) |                [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_test.json)                      |   |      |   |
| Synthtext |   | [link](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) |   | [instances_training.json](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.json) [instances_training.txt](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.txt)|-| | | |

- For `icdar2015`:
  - Step1: Download `ch4_training_images.zip` and `ch4_test_images.zip` from this [link](https://rrc.cvc.uab.es/?ch=4&com=downloads)
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

## Text Recognition
**The structure of the text recognition dataset directory is organized as follows.**

```
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
│   ├── Synth90k
│   │   ├── shuffle_labels.txt
│   │   ├── mnt
│   ├── SynthText
│   │   ├── shuffle_labels.txt
│   │   ├── instances_train.txt
│   │   ├── synthtext
│   ├── SynthAdd
│   │   ├── label.txt
│   │   ├── SynthText_Add

```
|   Dataset  |   |                                       images                                      |                                            annotation file                                           |                                             annotation file                                             | Note |
|:----------:|:-:|:---------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:----:|
||   | |training | test |      |
| coco_text ||[link](https://rrc.cvc.uab.es/?ch=5&com=downloads) |[train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_label.txt) |- |      |
| icdar_2011 ||[link](http://www.cvc.uab.es/icdar2011competition/?com=downloads) |[train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt) |- |      |
| icdar_2013 |   | [link](https://rrc.cvc.uab.es/?ch=2&com=downloads)                                | [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/train_label.txt)      | [test_label_1015.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/test_label_1015.txt) |      |
| icdar_2015 |   | [link](https://rrc.cvc.uab.es/?ch=4&com=downloads)                                | [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt)      | [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt)           |      |
| IIIT5K     |   | [link](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)        | [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/train_label.txt)          | [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)               |      |
| ct80       |   | - |-|[test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/ct80/test_label.txt)||
| svt        |   | [link](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) | -                                                                                                    | [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svt/test_label.txt)                  |      |
| svtp        |   | - | -                                                                                                    | [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svtp/test_label.txt)                  |      |
| Synth90k   |   | [link](https://www.robots.ox.ac.uk/~vgg/data/text/)                               | [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Synth90k/shuffle_labels.txt)  | -  |      |
| SynthText  |   | [link](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)                          | [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/shuffle_labels.txt) &#124; [instances_train.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/instances_train.txt) |    -  |      |
| SynthAdd   |   |       [link](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/SynthText_Add.zip)                                                                            |   [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/label.txt)|- |      |

- For `icdar_2013`:
  - Step1: Download `Challenge2_Test_Task3_Images.zip` and `Challenge2_Training_Task3_Images_GT.zip` from this [link](https://rrc.cvc.uab.es/?ch=2&com=downloads)
  - Step2: Download [test_label_1015.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/test_label_1015.txt) and [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/train_label.txt)
- For `icdar_2015`:
  - Step1: Download `ch4_training_word_images_gt.zip` and `ch4_test_word_images_gt.zip` from this [link](https://rrc.cvc.uab.es/?ch=4&com=downloads)
  - Step2: Download [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt) and [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt)
- For `IIIT5K`:
  - Step1: Download `IIIT5K-Word_V3.0.tar.gz` from this [link](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)
  - Step2: Download [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/train_label.txt) and [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)
- For `svt`:
  - Step1: Download `svt.zip` form this [link](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
  - Step2: Download [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svt/test_label.txt)
- For `ct80`:
  - Step1: Download [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/ct80/test_label.txt)
- For `svtp`:
  - Step1: Download [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svtp/test_label.txt)
- For `coco_text`:
  - Step1: Download from this [link](https://rrc.cvc.uab.es/?ch=5&com=downloads)
  - Step2: Download [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_label.txt)

- For `Syn90k`:
  - Step1: Download `mjsynth.tar.gz` from this [link](https://www.robots.ox.ac.uk/~vgg/data/text/)
  - Step2: Download [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Synth90k/shuffle_labels.txt)
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
  - Step1: Download `SynthText.zip` from this [link](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
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
  - Step1: Download `SynthText_Add.zip` from this [link](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/SynthText_Add.zip)
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
