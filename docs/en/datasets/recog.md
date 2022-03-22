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
│   ├── BID
│   │   ├── imgs
│   │   ├── dst_imgs
│   │   ├── annotations
│   │   ├── train_label.txt
│   │   ├── test_label.txt
```

|        Dataset        |                                                images                                                 |                                                                                                                                                                                                    annotation file                                                                                                                                                                                                    |                                                      annotation file                                                      |
| :-------------------: | :---------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
|                       |                                                                                                       |                                                                                                                                                                                                       training                                                                                                                                                                                                        |                                                           test                                                            |
|       coco_text       |                        [homepage](https://rrc.cvc.uab.es/?ch=5&com=downloads)                         |                                                                                                                                                            [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_label.txt)                                                                                                                                                             |                                                             -                                                             |  |
|      icdar_2011       |                 [homepage](http://www.cvc.uab.es/icdar2011competition/?com=downloads)                 |                                                                                                                                                            [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt)                                                                                                                                                            |                                                             -                                                             |  |
|      icdar_2013       |                        [homepage](https://rrc.cvc.uab.es/?ch=2&com=downloads)                         |                                                                                                                                                            [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/train_label.txt)                                                                                                                                                            |          [test_label_1015.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/test_label_1015.txt)          |  |
|      icdar_2015       |                        [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)                         |                                                                                                                                                            [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt)                                                                                                                                                            |               [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt)               |  |
|        IIIT5K         |            [homepage](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)             |                                                                                                                                                              [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/train_label.txt)                                                                                                                                                              |                 [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)                 |  |
|         ct80          |                     [homepage](http://cs-chan.com/downloads_CUTE80_dataset.html)                      |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                  [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/ct80/test_label.txt)                  |  |
|          svt          |         [homepage](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)         |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                  [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svt/test_label.txt)                   |  |
|         svtp          | [unofficial homepage\[1\]](https://github.com/Jyouhou/Case-Sensitive-Scene-Text-Recognition-Datasets) |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                  [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svtp/test_label.txt)                  |  |
|   MJSynth (Syn90k)    |                        [homepage](https://www.robots.ox.ac.uk/~vgg/data/text/)                        |                                                                                                                 [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/shuffle_labels.txt) \| [label.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/label.txt)                                                                                                                  |                                                             -                                                             |  |
| SynthText (Synth800k) |                     [homepage](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)                      | [alphanumeric_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/alphanumeric_labels.txt) \|[shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/shuffle_labels.txt) \| [instances_train.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/instances_train.txt) \| [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/label.txt) |                                                             -                                                             |  |
|       SynthAdd        |           [SynthText_Add.zip](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg)  (code:627x)           |                                                                                                                                                                   [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/label.txt)                                                                                                                                                                   |                                                             -                                                             |  |
|        TextOCR        |                            [homepage](https://textvqa.org/textocr/dataset)                            |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             |  |
|       Totaltext       |                       [homepage](https://github.com/cs-chan/Total-Text-Dataset)                       |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             |  |
|       OpenVINO        |                  [Open Images](https://github.com/cvdfoundation/open-images-dataset)                  |                                                                                                                                               [annotations](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text)                                                                                                                                               | [annotations](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text) |  |
|         FUNSD         |                          [homepage](https://guillaumejaume.github.io/FUNSD/)                          |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             |  |
|          BID          |          [homepage](https://drive.google.com/file/d/1Oi88TRcpdjZmJ79WDLb9qFlBNG8q2De6/view)           |                                                                                                                                                                                                           -                                                                                                                                                                                                           |                                                             -                                                             |  |


(*) Since the official homepage is unavailable now, we provide an alternative for quick reference. However, we do not guarantee the correctness of the dataset.

## Preparation Steps

### ICDAR 2013
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


### BID

- Step1: Download [dataset.zip](https://drive.google.com/file/d/1Oi88TRcpdjZmJ79WDLb9qFlBNG8q2De6/view)
- Step2: Run the following commands to preprocess the dataset
```bash
mv BID\ Dataset.zip BID_Dataset.zip
mkdir data && mv BID.zip data/
cd data
unzip -q BID_Dataset.zip && rm BID_Dataset.zip
mv BID\ Dataset BID_Dataset
cd BID_Dataset
mv CNH_Aberta/*in.jpg imgs && mv CNH_Aberta/*txt annotations && rm -rf CNH_Aberta
mv CNH_Frente/*in.jpg imgs && mv CNH_Frente/*txt annotations && rm -rf CNH_Frente
mv CNH_Verso/*in.jpg imgs && mv CNH_Verso/*txt annotations && rm -rf CNH_Verso
mv CPF_Frente/*in.jpg imgs && mv CPF_Frente/*txt annotations && rm -rf CPF_Frente
mv CPF_Verso/*in.jpg imgs && mv CPF_Verso/*txt annotations && rm -rf CPF_Verso
mv RG_Aberto/*in.jpg imgs && mv RG_Aberto/*txt annotations && rm -rf RG_Aberto
mv RG_Frente/*in.jpg imgs && mv RG_Frente/*txt annotations && rm -rf RG_Frente
mv RG_Verso/*in.jpg imgs && mv RG_Verso/*txt annotations && rm -rf RG_Verso
cd ../../
```
- Step3: Generate `train_label.txt` and `test_label.txt` and crop images using 4 processes with the following command (add `--preserve_vertical` if you wish to preserve the images containing vertical texts). Since the original dataset doesn't have a test set, you may specify `--test_ratio` to split the dataset. E.g., if test_ratio is 0.2, then 20% of the data are left out as the test set in this example.
```bash
python tools/data/textrecog/bid_converter.py data/BID_Dataset --nproc 4 --test_ratio 0.2
```