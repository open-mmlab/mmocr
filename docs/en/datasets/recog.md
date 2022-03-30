# Text Recognition

## Overview

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
│   │   ├── alphanumeric_labels.txt
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
│   ├── OpenVINO
│   │   ├── image_1
│   │   ├── image_2
│   │   ├── image_5
│   │   ├── image_f
│   │   ├── image_val
│   │   ├── train_1_label.txt
│   │   ├── train_2_label.txt
│   │   ├── train_5_label.txt
│   │   ├── train_f_label.txt
│   │   ├── val_label.txt
│   ├── funsd
│   │   ├── imgs
│   │   ├── dst_imgs
│   │   ├── annotations
│   │   ├── train_label.txt
│   │   ├── test_label.txt
│   ├── lv
│   │   ├── Crops
│   │   ├── train_label.jsonl
│   │   ├── test_label.jsonl
```

|        Dataset        |                                                images                                                 |                                                                                                                                                                                                    annotation file                                                                                                                                                                                                    |                                                      annotation file                                                      |
| :-------------------: | :---------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
|                       |                                                                                                       |                                                                                                                                                                                                       training                                                                                                                                                                                                        |                                                           test                                                            |
|       coco_text       |                        [homepage](https://rrc.cvc.uab.es/?ch=5&com=downloads)                         |                                                                                                                                                            [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_label.txt)                                                                                                                                                             |                                                             -                                                             |   |
|       ICDAR2011       |                               [homepage](https://rrc.cvc.uab.es/?ch=1)                                |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             |   |
|       ICDAR2013       |                               [homepage](https://rrc.cvc.uab.es/?ch=2)                                |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             | - |
|      icdar_2015       |                        [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)                         |                                                                                                                                                            [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt)                                                                                                                                                            |               [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt)               |   |
|        IIIT5K         |            [homepage](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)             |                                                                                                                                                              [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/train_label.txt)                                                                                                                                                              |                 [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)                 |   |
|         ct80          |                     [homepage](http://cs-chan.com/downloads_CUTE80_dataset.html)                      |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                  [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/ct80/test_label.txt)                  |   |
|          svt          |         [homepage](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)         |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                  [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svt/test_label.txt)                   |   |
|         svtp          | [unofficial homepage\[1\]](https://github.com/Jyouhou/Case-Sensitive-Scene-Text-Recognition-Datasets) |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                  [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svtp/test_label.txt)                  |   |
|   MJSynth (Syn90k)    |                        [homepage](https://www.robots.ox.ac.uk/~vgg/data/text/)                        |                                                                                                                 [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/shuffle_labels.txt) \| [label.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/label.txt)                                                                                                                  |                                                             -                                                             |   |
| SynthText (Synth800k) |                     [homepage](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)                      | [alphanumeric_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/alphanumeric_labels.txt) \|[shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/shuffle_labels.txt) \| [instances_train.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/instances_train.txt) \| [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/label.txt) |                                                             -                                                             |   |
|       SynthAdd        |           [SynthText_Add.zip](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg)  (code:627x)           |                                                                                                                                                                   [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/label.txt)                                                                                                                                                                   |                                                             -                                                             |   |
|        TextOCR        |                            [homepage](https://textvqa.org/textocr/dataset)                            |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             |   |
|       Totaltext       |                       [homepage](https://github.com/cs-chan/Total-Text-Dataset)                       |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             |   |
|       OpenVINO        |                  [Open Images](https://github.com/cvdfoundation/open-images-dataset)                  |                                                                                                                                               [annotations](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text)                                                                                                                                               | [annotations](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text) |   |
|         FUNSD         |                          [homepage](https://guillaumejaume.github.io/FUNSD/)                          |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             |   |
|        DeText         |                               [homepage](https://rrc.cvc.uab.es/?ch=9)                                |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             |   |
|          NAF          |                           [homepage](https://github.com/herobd/NAF_dataset)                           |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             | - |
|         SROIE         |                               [homepage](https://rrc.cvc.uab.es/?ch=13)                               |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             | - |
|   Lecture Video DB    |          [homepage](https://cvit.iiit.ac.in/research/projects/cvit-projects/lecturevideodb)           |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             | - |
|         IMGUR         |              [homepage](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset)              |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             | - |
|         KAIST         |          [homepage](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database)           |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             | - |
|         MTWI          |       [homepage](https://tianchi.aliyun.com/competition/entrance/231685/information?lang=en-us)       |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             | - |
|     COCO Text v2      |                            [homepage](https://bgshih.github.io/cocotext/)                             |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             | - |
|         ReCTS         |                               [homepage](https://rrc.cvc.uab.es/?ch=12)                               |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             | - |


(*) Since the official homepage is unavailable now, we provide an alternative for quick reference. However, we do not guarantee the correctness of the dataset.

## Preparation Steps

### ICDAR 2011 (Born-Digital Images)
- Step1: Download `Challenge1_Training_Task3_Images_GT.zip`, `Challenge1_Test_Task3_Images.zip`, and `Challenge1_Test_Task3_GT.txt` from [homepage](https://rrc.cvc.uab.es/?ch=1&com=downloads) `Task 1.3: Word Recognition (2013 edition)`.

  ```bash
  mkdir icdar2011 && cd icdar2011
  mkdir annotations

  # Download ICDAR 2011
  wget https://rrc.cvc.uab.es/downloads/Challenge1_Training_Task3_Images_GT.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge1_Test_Task3_Images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge1_Test_Task3_GT.txt --no-check-certificate

  # For images
  mkdir crops
  unzip -q Challenge1_Training_Task3_Images_GT.zip -d crops/train
  unzip -q Challenge1_Test_Task3_Images.zip -d crops/test

  # For annotations
  mv Challenge1_Test_Task3_GT.txt annotations && mv train/gt.txt annotations/Challenge1_Train_Task3_GT.txt
  ```

- Step2: Convert original annotations to `Train_label.jsonl` and `Test_label.jsonl` with the following command:

  ```bash
  python tools/data/textrecog/ic11_converter.py PATH/TO/icdar2011
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── icdar2011
  │   ├── crops
  │   ├── train_label.jsonl
  │   └── test_label.jsonl
  ```

### ICDAR 2013 (Focused Scene Text)
- Step1: Download `Challenge2_Training_Task3_Images_GT.zip`, `Challenge2_Test_Task3_Images.zip`, and `Challenge2_Test_Task3_GT.txt` from [homepage](https://rrc.cvc.uab.es/?ch=2&com=downloads) `Task 2.3: Word Recognition (2013 edition)`.

  ```bash
  mkdir icdar2013 && cd icdar2013
  mkdir annotations

  # Download ICDAR 2013
  wget https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task3_Images_GT.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task3_Images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task3_GT.txt --no-check-certificate

  # For images
  mkdir crops
  unzip -q Challenge2_Training_Task3_Images_GT.zip -d crops/train
  unzip -q Challenge2_Test_Task3_Images.zip -d crops/test
  # For annotations
  mv Challenge2_Test_Task3_GT.txt annotations && mv crops/train/gt.txt annotations/Challenge2_Train_Task3_GT.txt

  rm Challenge2_Training_Task3_Images_GT.zip && rm Challenge2_Test_Task3_Images.zip
  ```

- Step 2: Generate `Train_label.jsonl` and `Test_label.jsonl` with the following command:

  ```bash
  python tools/data/textrecog/ic13_converter.py PATH/TO/icdar2013
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── icdar2013
  │   ├── crops
  │   ├── train_label.jsonl
  │   └── test_label.jsonl
  ```

### ICDAR 2013 [Deprecated]
- Step1: Download `Challenge2_Test_Task3_Images.zip` and `Challenge2_Training_Task3_Images_GT.zip` from [homepage](https://rrc.cvc.uab.es/?ch=2&com=downloads)
- Step2: Download [test_label_1015.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/test_label_1015.txt) and [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/train_label.txt)

### ICDAR 2015
- Step1: Download `ch4_training_word_images_gt.zip` and `ch4_test_word_images_gt.zip` from [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)
- Step2: Download [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt) and [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt)
### IIIT5K
  - Step1: Download `IIIT5K-Word_V3.0.tar.gz` from [homepage](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)
  - Step2: Download [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/train_label.txt) and [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)

### svt
  - Step1: Download `svt.zip` form [homepage](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
  - Step2: Download [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svt/test_label.txt)
  - Step3:
  ```bash
  python tools/data/textrecog/svt_converter.py <download_svt_dir_path>
  ```

### ct80
  - Step1: Download [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/ct80/test_label.txt)

### svtp
  - Step1: Download [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svtp/test_label.txt)

### coco_text
  - Step1: Download from [homepage](https://rrc.cvc.uab.es/?ch=5&com=downloads)
  - Step2: Download [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_label.txt)

### MJSynth (Syn90k)
  - Step1: Download `mjsynth.tar.gz` from [homepage](https://www.robots.ox.ac.uk/~vgg/data/text/)
  - Step2: Download [label.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/label.txt) (8,919,273 annotations) and [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/shuffle_labels.txt) (2,400,000 randomly sampled annotations). **Please make sure you're using the right annotation to train the model by checking its dataset specs in Model Zoo.**
  - Step3:

  ```bash
  mkdir Syn90k && cd Syn90k

  mv /path/to/mjsynth.tar.gz .

  tar -xzf mjsynth.tar.gz

  mv /path/to/shuffle_labels.txt .
  mv /path/to/label.txt .

  # create soft link
  cd /path/to/mmocr/data/mixture

  ln -s /path/to/Syn90k Syn90k
  ```

### SynthText (Synth800k)
- Step1: Download `SynthText.zip` from [homepage](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)

- Step2: According to your actual needs, download the most appropriate one from the following options: [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/label.txt) (7,266,686 annotations), [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/shuffle_labels.txt) (2,400,000 randomly sampled annotations), [alphanumeric_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/alphanumeric_labels.txt) (7,239,272 annotations with alphanumeric characters only) and [instances_train.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/instances_train.txt) (7,266,686 character-level annotations).

:::{warning}
Please make sure you're using the right annotation to train the model by checking its dataset specs in Model Zoo.
:::

- Step3:

```bash
mkdir SynthText && cd SynthText
mv /path/to/SynthText.zip .
unzip SynthText.zip
mv SynthText synthtext

mv /path/to/shuffle_labels.txt .
mv /path/to/label.txt .
mv /path/to/alphanumeric_labels.txt .
mv /path/to/instances_train.txt .

# create soft link
cd /path/to/mmocr/data/mixture
ln -s /path/to/SynthText SynthText
```

- Step4:
Generate cropped images and labels:

```bash
cd /path/to/mmocr

python tools/data/textrecog/synthtext_converter.py data/mixture/SynthText/gt.mat data/mixture/SynthText/ data/mixture/SynthText/synthtext/SynthText_patch_horizontal --n_proc 8
```

### SynthAdd
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
:::{tip}
To convert label file with `txt` format to `lmdb` format,
```bash
python tools/data/utils/txt2lmdb.py -i <txt_label_path> -o <lmdb_label_path>
```
For example,
```bash
python tools/data/utils/txt2lmdb.py -i data/mixture/Syn90k/label.txt -o data/mixture/Syn90k/label.lmdb
```
:::

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
  - Step2: Generate `train_label.txt`, `val_label.txt` and crop images using 4 processes with the following command:
  ```bash
  python tools/data/textrecog/textocr_converter.py /path/to/textocr 4
  ```

### Totaltext
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
  - Step2: Generate cropped images, `train_label.txt` and `test_label.txt` with the following command (the cropped images will be saved to `data/totaltext/dst_imgs/`):
  ```bash
  python tools/data/textrecog/totaltext_converter.py /path/to/totaltext -o /path/to/totaltext --split-list training test
  ```

### OpenVINO
  - Step0: Install [awscli](https://aws.amazon.com/cli/).
  - Step1: Download [Open Images](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations) subsets `train_1`, `train_2`, `train_5`, `train_f`, and `validation` to `openvino/`.
  ```bash
  mkdir openvino && cd openvino

  # Download Open Images subsets
  for s in 1 2 5 f; do
    aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_${s}.tar.gz .
  done
  aws s3 --no-sign-request cp s3://open-images-dataset/tar/validation.tar.gz .

  # Download annotations
  for s in 1 2 5 f; do
    wget https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text/text_spotting_openimages_v5_train_${s}.json
  done
  wget https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text/text_spotting_openimages_v5_validation.json

  # Extract images
  mkdir -p openimages_v5/val
  for s in 1 2 5 f; do
    tar zxf train_${s}.tar.gz -C openimages_v5
  done
  tar zxf validation.tar.gz -C openimages_v5/val
  ```
  - Step2: Generate `train_{1,2,5,f}_label.txt`, `val_label.txt` and crop images using 4 processes with the following command:
  ```bash
  python tools/data/textrecog/openvino_converter.py /path/to/openvino 4
  ```

### DeText

- Step1: Download `ch9_training_images.zip`, `ch9_training_localization_transcription_gt.zip`, `ch9_validation_images.zip`, and `ch9_validation_localization_transcription_gt.zip` from **Task 3: End to End** on the [homepage](https://rrc.cvc.uab.es/?ch=9).

  ```bash
  mkdir detext && cd detext
  mkdir imgs && mkdir annotations && mkdir imgs/training && mkdir imgs/val && mkdir annotations/training && mkdir annotations/val

  # Download DeText
  wget https://rrc.cvc.uab.es/downloads/ch9_training_images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch9_training_localization_transcription_gt.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch9_validation_images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch9_validation_localization_transcription_gt.zip --no-check-certificate

  # Extract images and annotations
  unzip -q ch9_training_images.zip -d imgs/training && unzip -q ch9_training_localization_transcription_gt.zip -d annotations/training && unzip -q ch9_validation_images.zip -d imgs/val && unzip -q ch9_validation_localization_transcription_gt.zip -d annotations/val

  # Remove zips
  rm ch9_training_images.zip && rm ch9_training_localization_transcription_gt.zip && rm ch9_validation_images.zip && rm ch9_validation_localization_transcription_gt.zip
  ```

- Step2: Generate `instances_training.json` and `instances_val.json` with following command:

  ```bash
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/detext/ignores
  python tools/data/textrecog/detext_converter.py PATH/TO/detext --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── detext
  │   ├── crops
  │   ├── ignores
  │   ├── train_label.jsonl
  │   ├── test_label.jsonl
  ```

### NAF

- Step1: Download [labeled_images.tar.gz](https://github.com/herobd/NAF_dataset/releases/tag/v1.0) to `naf/`.

  ```bash
  mkdir naf && cd naf

  # Download NAF dataset
  wget https://github.com/herobd/NAF_dataset/releases/download/v1.0/labeled_images.tar.gz
  tar -zxf labeled_images.tar.gz

  # For images
  mkdir annotations && mv labeled_images imgs

  # For annotations
  git clone https://github.com/herobd/NAF_dataset.git
  mv NAF_dataset/train_valid_test_split.json annotations/ && mv NAF_dataset/groups annotations/

  rm -rf NAF_dataset && rm labeled_images.tar.gz
  ```

- Step2: Generate `train_label.txt`, `val_label.txt`, and `test_label.txt` with following command:

  ```bash
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/naf/ignores
  python tools/data/textrecog/naf_converter.py PATH/TO/naf --nproc 4

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── naf
  │   ├── crops
  │   ├── train_label.txt
  │   ├── val_label.txt
  │   ├── test_label.txt
  ```
### SROIE

- Step1: Step1: Download `0325updated.task1train(626p).zip`, `task1&2_test(361p).zip`, and `text.task1&2-test（361p).zip` from [homepage](https://rrc.cvc.uab.es/?ch=13&com=downloads) to `sroie/`

- Step2:

  ```bash
  mkdir sroie && cd sroie
  mkdir imgs && mkdir annotations && mkdir imgs/training

  # Warnninig: The zip files downloaded from Google Drive and BaiduYun Cloud may
  # be different, the user should revise the following commands to the correct
  # file name if encounter with errors while extracting and move the files.
  unzip -q 0325updated.task1train\(626p\).zip && unzip -q task1\&2_test\(361p\).zip && unzip -q text.task1\&2-test（361p\).zip

  # For images
  mv 0325updated.task1train\(626p\)/*.jpg imgs/training && mv fulltext_test\(361p\) imgs/test

  # For annotations
  mv 0325updated.task1train\(626p\) annotations/training && mv text.task1\&2-testги361p\)/ annotations/test

  rm 0325updated.task1train\(626p\).zip && rm task1\&2_test\(361p\).zip && rm text.task1\&2-test（361p\).zip
  ```

- Step3: Generate `train_label.jsonl` and `test_label.jsonl` and crop images using 4 processes with the following command:

  ```bash
  python tools/data/textrecog/sroie_converter.py PATH/TO/sroie --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── sroie
  │   ├── crops
  │   ├── train_label.jsonl
  │   ├── test_label.jsonl
  ```
### Lecture Video DB

**The LV dataset has already provided cropped images and the corresponding annotations**

- Step1: Download [IIIT-CVid.zip](http://cdn.iiit.ac.in/cdn/preon.iiit.ac.in/~kartik/IIIT-CVid.zip) to `lv/`.

```bash
mkdir lv && cd lv

# Download LV dataset
wget http://cdn.iiit.ac.in/cdn/preon.iiit.ac.in/~kartik/IIIT-CVid.zip
unzip -q IIIT-CVid.zip

# For image
mv IIIT-CVid/Crops ./

# For annotation
mv IIIT-CVid/train.txt train_label.txt && mv IIIT-CVid/val.txt val_label.txt && mv IIIT-CVid/test.txt test_label.txt

rm IIIT-CVid.zip
```

- Step2: Generate `train_label.jsonl`, `val.jsonl`, and `test.jsonl` with following command:

```bash
python tools/data/textdreog/lv_converter.py PATH/TO/lv
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

- Step2: Generate `train_label.txt` and `test_label.txt` and crop images using 4 processes with following command (add `--preserve-vertical` if you wish to preserve the images containing vertical texts):

```bash
python tools/data/textrecog/funsd_converter.py PATH/TO/funsd --nproc 4
```

### IMGUR

- Step1: Run `download_imgur5k.py` to download images. You can merge [PR#5](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset/pull/5) in your local repository to enable a **much faster** parallel execution of image download.

  ```bash
  mkdir imgur && cd imgur

  git clone https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset.git

  # Download images from imgur.com. This may take SEVERAL HOURS!
  python ./IMGUR5K-Handwriting-Dataset/download_imgur5k.py --dataset_info_dir ./IMGUR5K-Handwriting-Dataset/dataset_info/ --output_dir ./imgs

  # For annotations
  mkdir annotations
  mv ./IMGUR5K-Handwriting-Dataset/dataset_info/*.json annotations

  rm -rf IMGUR5K-Handwriting-Dataset
  ```

- Step2: Generate `train_label.txt`, `val_label.txt` and `test_label.txt` and crop images with the following command:

  ```bash
  python tools/data/textrecog/imgur_converter.py PATH/TO/imgur
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── imgur
  │   ├── crops
  │   ├── train_label.jsonl
  │   ├── test_label.jsonl
  │   └── val_label.jsonl
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

- Step2: Generate `train_label.txt` and `test_label.txt` and crop images using 4 processes with following command (add `--preserve-vertical` if you wish to preserve the images containing vertical texts):

```bash
python tools/data/textrecog/funsd_converter.py PATH/TO/funsd --nproc 4
```

### KAIST

- Step1: Complete download [KAIST_all.zip](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database) to `kaist/`.

  ```bash
  mkdir kaist && cd kaist
  mkdir imgs && mkdir annotations

  # Download KAIST dataset
  wget http://www.iapr-tc11.org/dataset/KAIST_SceneText/KAIST_all.zip
  unzip -q KAIST_all.zip

  rm KAIST_all.zip
  ```

- Step2: Extract zips:

  ```bash
  python tools/data/common/extract_kaist.py PATH/TO/kaist
  ```

- Step3: Generate `train_label.jsonl` and `val_label.jsonl` (optional) with following command:

  ```bash
  # Since KAIST does not provide an official split, you can split the dataset by adding --val-ratio 0.2
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/kaist/ignores
  python tools/data/textrecog/kaist_converter.py PATH/TO/kaist --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── kaist
  │   ├── crops
  │   ├── ignores
  │   ├── train_label.jsonl
  │   └── val_label.jsonl (optional)
  ```

### MTWI

- Step1: Download `mtwi_2018_train.zip` from [homepage](https://tianchi.aliyun.com/competition/entrance/231685/information?lang=en-us).

  ```bash
  mkdir mtwi && cd mtwi

  unzip -q mtwi_2018_train.zip
  mv image_train imgs && mv txt_train annotations

  rm mtwi_2018_train.zip
  ```

- Step2: Generate `train_label.jsonl` and `val_label.jsonl` (optional) with the following command:

  ```bash
  # Annotations of MTWI test split is not publicly available, split a validation
  # set by adding --val-ratio 0.2
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/mtwi/ignores
  python tools/data/textrecog/mtwi_converter.py PATH/TO/mtwi --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── mtwi
  │   ├── crops
  │   ├── train_label.jsonl
  │   ├── val_label.jsonl (optional)
  ```

### COCO Text v2

- Step1: Download image [train2014.zip](http://images.cocodataset.org/zips/train2014.zip) and annotation [cocotext.v2.zip](https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip) to `coco_textv2/`.

  ```bash
  mkdir coco_textv2 && cd coco_textv2
  mkdir annotations

  # Download COCO Text v2 dataset
  wget http://images.cocodataset.org/zips/train2014.zip
  wget https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip
  unzip -q train2014.zip && unzip -q cocotext.v2.zip

  mv train2014 imgs && mv cocotext.v2.json annotations/

  rm train2014.zip && rm -rf cocotext.v2.zip
  ```

- Step2: Generate `train_label.jsonl` and `val_label.jsonl` with the following command:

  ```bash
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/mtwi/ignores
  python tools/data/textrecog/cocotext_converter.py PATH/TO/coco_textv2 --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── coco_textv2
  │   ├── crops
  │   ├── ignores
  │   ├── train_label.jsonl
  │   └── val_label.jsonl
  ```

### ReCTS

- Step1: Download [ReCTS.zip](https://datasets.cvc.uab.es/rrc/ReCTS.zip) to `rects/` from the [homepage](https://rrc.cvc.uab.es/?ch=12&com=downloads).

  ```bash
  mkdir rects && cd rects

  # Download ReCTS dataset
  # You can also find Google Drive link on the dataset homepage
  wget https://datasets.cvc.uab.es/rrc/ReCTS.zip --no-check-certificate
  unzip -q ReCTS.zip

  mv img imgs && mv gt_unicode annotations

  rm ReCTS.zip -f && rm -rf gt
  ```

- Step2: Generate `train_label.jsonl` and `val_label.jsonl` (optional) with the following command:

  ```bash
  # Annotations of ReCTS test split is not publicly available, split a validation
  # set by adding --val-ratio 0.2
  # Add --preserve-vertical to preserve vertical texts for training, otherwise
  # vertical images will be filtered and stored in PATH/TO/rects/ignores
  python tools/data/textrecog/rects_converter.py PATH/TO/rects --nproc 4
  ```

- After running the above codes, the directory structure should be as follows:

  ```text
  ├── rects
  │   ├── crops
  │   ├── ignores
  │   ├── train_label.jsonl
  │   └── val_label.jsonl (optional)
  ```
