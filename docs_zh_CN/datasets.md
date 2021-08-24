# 配置数据集

本页列出了在文字检测、文字识别、关键信息提取、命名实体识别四个文本类任务中常用的数据集以及下载链接。

## 文字检测

文字检测任务的数据集应按如下目录配置：

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

|  数据集名称  |                             数据图片                             |                                                                                                     |               标注文件                  |                                                                                                |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :-------------------------------------: | :--------------------------------------------------------------------------------------------: |
|           |                                                                                      |                                                训练集 (training)                                                |               验证集 (validation)                |                                           测试集 (testing)                                             |       |
|  CTW1500  | [下载地址](https://github.com/Yuliang-Liu/Curve-Text-Detector) |                    -                    |                    -                    |                    -                    |
| ICDAR2015 | [下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads)     | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json) |                    -                    | [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json) |
| ICDAR2017 | [下载地址](https://rrc.cvc.uab.es/?ch=8&com=downloads)     | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_training.json) | [instances_val.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_val.json) | - |       |       |
| Synthtext | [下载地址](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)  | [instances_training.lmdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb) |                    -                    | - |
| TextOCR | [下载地址](https://textvqa.org/textocr/dataset)  | - |                    -                    | -
| Totaltext | [下载地址](https://github.com/cs-chan/Total-Text-Dataset)  | - |                    -                    | -

**注意：若用户需要在 CTW1500, ICDAR 2015/2017 或 Totaltext 数据集上训练模型**, 请注意这些数据集中有部分图片的 EXIF 信息里保存着方向信息。MMCV 采用的 OpenCV 后端会默认根据方向信息对图片进行旋转；而由于数据集的标注是在原图片上进行的，这种冲突会使得部分训练样本失效。因此，用户应该在配置 pipeline 时使用 `dict(type='mmdet.LoadImageFromFile', color_type='color_ignore_orientation')` 以避免 MMCV 的这一行为。（配置文件可参考 [DBNet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py)）

- `icdar2015` 数据集：
  - 第一步：从[下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载 `ch4_training_images.zip`、`ch4_test_images.zip`、`ch4_training_localization_transcription_gt.zip`、`Challenge4_Test_Task1_GT.zip` 四个文件，分别对应训练集数据、测试集数据、训练集标注、测试集标注。
  - 第二步：运行以下命令，移动数据集到对应文件夹
  ```bash
  mkdir icdar2015 && cd icdar2015
  mkdir imgs && mkdir annotations
  # 移动数据到目录：
  mv ch4_training_images imgs/training
  mv ch4_test_images imgs/test
  # 移动标注到目录：
  mv ch4_training_localization_transcription_gt annotations/training
  mv Challenge4_Test_Task1_GT annotations/test
  ```
  - 第三步：下载 [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json) 和 [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json)，并放入 `icdar2015` 文件夹里。或者也可以用以下命令直接生成 `instances_training.json` 和 `instances_test.json`:
  ```bash
  python tools/data/textdet/icdar_converter.py /path/to/icdar2015 -o /path/to/icdar2015 -d icdar2015 --split-list training test
  ```

- `icdar2017` 数据集：
  - 与上述步骤类似。

- `ctw1500` 数据集：
  - 第一步：执行以下命令，从 [下载地址](https://github.com/Yuliang-Liu/Curve-Text-Detector) 下载 `train_images.zip`，`test_images.zip`，`train_labels.zip`，`test_labels.zip` 四个文件并配置到对应目录：

  ```bash
  mkdir ctw1500 && cd ctw1500
  mkdir imgs && mkdir annotations

  # 下载并配置标注
  cd annotations
  wget -O train_labels.zip https://universityofadelaide.box.com/shared/static/jikuazluzyj4lq6umzei7m2ppmt3afyw.zip
  wget -O test_labels.zip https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/download
  unzip train_labels.zip && mv ctw1500_train_labels training
  unzip test_labels.zip -d test
  cd ..
  # 下载并配置数据
  cd imgs
  wget -O train_images.zip https://universityofadelaide.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip
  wget -O test_images.zip https://universityofadelaide.box.com/shared/static/t4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip
  unzip train_images.zip && mv train_images training
  unzip test_images.zip && mv test_images test
  ```
  - 第二步：执行以下命令，生成 `instances_training.json` 和 `instances_test.json`。

  ```bash
  python tools/data/textdet/ctw1500_converter.py /path/to/ctw1500 -o /path/to/ctw1500 --split-list training test
  ```

- `TextOCR` 数据集：
  - 第一步：下载 [train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)，[TextOCR_0.1_train.json](https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json) 和 [TextOCR_0.1_val.json](https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json) 到 `textocr` 文件夹里。
  ```bash
  mkdir textocr && cd textocr

  # 下载 TextOCR 数据集
  wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
  wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json
  wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json

  # 把图片移到对应目录
  unzip -q train_val_images.zip
  mv train_images train
  ```

  - 第二步：生成 `instances_training.json` 和 `instances_val.json`:
  ```bash
  python tools/data/textdet/textocr_converter.py /path/to/textocr
  ```

- `Totaltext` 数据集：
  - 第一步：从 [github dataset](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) 下载 `totaltext.zip`，从 [github Groundtruth](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Groundtruth/Text) 下载 `groundtruth_text.zip` 。（建议下载 `.mat` 格式的标注文件，因为我们提供的标注格式转换脚本 `totaltext_converter.py` 仅支持 `.mat` 格式。）
  ```bash
  mkdir totaltext && cd totaltext
  mkdir imgs && mkdir annotations

  # 图像
  # 在 ./totaltext 中执行
  unzip totaltext.zip
  mv Images/Train imgs/training
  mv Images/Test imgs/test

  # 标注文件
  unzip groundtruth_text.zip
  cd Groundtruth
  mv Polygon/Train ../annotations/training
  mv Polygon/Test ../annotations/test

  ```
  - 第二步：用以下命令生成 `instances_training.json` 和 `instances_test.json` ：
  ```bash
  python tools/data/textdet/totaltext_converter.py /path/to/totaltext -o /path/to/totaltext --split-list training test
  ```

## 文字识别

**文字识别任务的数据集应按如下目录配置：**

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

|  数据集名称   |                                        数据图片                                         |                                                                                                                                            标注文件                                                                                                                                                 |                                             标注文件                                             |
| :--------: | :-----------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: |
|       |                                                                                       |                                                                                                                                                训练集(training)                                                                                                                                               |                                                  测试集(test)                                                   |
| coco_text  |                [下载地址](https://rrc.cvc.uab.es/?ch=5&com=downloads)                 |                                                                                                     [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_label.txt)                                                                                                     |                                                    -                                                    |       |
| icdar_2011 | [下载地址](http://www.cvc.uab.es/icdar2011competition/?com=downloads)         |                                                                                                    [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt)                                                                                                     |                                                    -                                                    |       |
| icdar_2013 |              [下载地址](https://rrc.cvc.uab.es/?ch=2&com=downloads)                 |                                                                                                    [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/train_label.txt)                                                                                                     | [test_label_1015.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/test_label_1015.txt) |       |
| icdar_2015 |               [下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads)                 |                                                                                                    [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt)                                                                                                     |      [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt)      |       |
|   IIIT5K   |    [下载地址](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)     |                                                                                                      [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/train_label.txt)                                                                                                       |        [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)        |       |
|    ct80    |                                            [下载地址](http://cs-chan.com/downloads_CUTE80_dataset.html)                                           |                                                                                                                                                   -                                                                                                                                                    |         [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/ct80/test_label.txt)         |       |
|    svt     |[下载地址](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) |                                                                                                                                                   -                                                                                                                                                    |         [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svt/test_label.txt)          |       |
|    svtp    |                              [非官方下载地址*](https://github.com/Jyouhou/Case-Sensitive-Scene-Text-Recognition-Datasets)                                           |                                                                                                                                                   -                                                                                                                                                    |         [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svtp/test_label.txt)         |       |
|  Syn90k  |               [下载地址](https://www.robots.ox.ac.uk/~vgg/data/text/)                |                                                       [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/shuffle_labels.txt) \| [label.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/label.txt)                                                       |                                                    -                                                    |       |
| SynthText  |           [下载地址](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)              | [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/shuffle_labels.txt) \| [instances_train.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/instances_train.txt) \| [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/label.txt) |                                                    -                                                    |       |
|  SynthAdd  |  [SynthText_Add.zip](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg)  (code:627x)   |                                                                                                           [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/label.txt)                                                                                                            |                                                    -                                                    |       |
|  TextOCR  |  [下载地址](https://textvqa.org/textocr/dataset)   |                                                                                                           -                                                                                                           |                                                    -                                                    |       |
|  Totaltext  |  [下载地址](https://github.com/cs-chan/Total-Text-Dataset)   |                                                                                                           -                                                                                                           |                                                    -                                                    |       |

(*) 注：由于官方的下载地址已经无法访问，我们提供了一个非官方的地址以供参考，但我们无法保证数据的准确性。

- `icdar_2013` 数据集：
  - 第一步：从 [下载地址](https://rrc.cvc.uab.es/?ch=2&com=downloads) 下载 `Challenge2_Test_Task3_Images.zip` 和 `Challenge2_Training_Task3_Images_GT.zip`
  - 第二步：下载 [test_label_1015.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/test_label_1015.txt) 和 [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/train_label.txt)
- `icdar_2015` 数据集：
  - 第一步：从 [下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads) 下载 `ch4_training_word_images_gt.zip` 和 `ch4_test_word_images_gt.zip`
  - 第二步：下载 [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt) and [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt)
- `IIIT5K` 数据集：
  - 第一步：从 [下载地址](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) 下载 `IIIT5K-Word_V3.0.tar.gz`
  - 第二步：下载 [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/train_label.txt) 和 [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)
- `svt` 数据集：
  - 第一步：从 [下载地址](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) 下载 `svt.zip`
  - 第二步：下载 [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svt/test_label.txt)
  - 第三步：
  ```bash
  python tools/data/textrecog/svt_converter.py <download_svt_dir_path>
  ```
- `ct80` 数据集：
  - 第一步：下载 [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/ct80/test_label.txt)
- `svtp` 数据集：
  - 第一步：下载 [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svtp/test_label.txt)
- `coco_text` 数据集：
  - 第一步：从 [下载地址](https://rrc.cvc.uab.es/?ch=5&com=downloads) 下载文件
  - 第二步：下载 [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_label.txt)
- `Syn90k` 数据集：
  - 第一步：从 [下载地址](https://www.robots.ox.ac.uk/~vgg/data/text/) 下载 `mjsynth.tar.gz`
  - 第二步：下载 [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/shuffle_labels.txt)
  - 第三步：

  ```bash
  mkdir Syn90k && cd Syn90k

  mv /path/to/mjsynth.tar.gz .

  tar -xzf mjsynth.tar.gz

  mv /path/to/shuffle_labels.txt .

  # 创建软链接
  cd /path/to/mmocr/data/mixture

  ln -s /path/to/Syn90k Syn90k
  ```

- `SynthText` 数据集：
  - 第一步： 从 [下载地址](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) 下载 `SynthText.zip`
  - 第二步：

  ```bash
  mkdir SynthText && cd SynthText
  mv /path/to/SynthText.zip .
  unzip SynthText.zip
  mv SynthText synthtext

  mv /path/to/shuffle_labels.txt .

  # 创建软链接
  cd /path/to/mmocr/data/mixture
  ln -s /path/to/SynthText SynthText
  ```
  - 第三步：
  生成裁剪后的图像和标注：

  ```bash
  cd /path/to/mmocr

  python tools/data/textrecog/synthtext_converter.py data/mixture/SynthText/gt.mat data/mixture/SynthText/ data/mixture/SynthText/synthtext/SynthText_patch_horizontal --n_proc 8
  ```

- `SynthAdd` 数据集：
  - 第一步：从 [SynthAdd](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg) (code:627x) 下载 `SynthText_Add.zip`
  - 第二步：下载 [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/label.txt)
  - 第三步：

  ```bash
  mkdir SynthAdd && cd SynthAdd

  mv /path/to/SynthText_Add.zip .

  unzip SynthText_Add.zip

  mv /path/to/label.txt .

  # 创建软链接
  cd /path/to/mmocr/data/mixture

  ln -s /path/to/SynthAdd SynthAdd
  ```
  **额外说明：**
运行以下命令，可以把 `.txt` 格式的标注文件转换成 `.lmdb` 格式：
```bash
python tools/data/utils/txt2lmdb.py -i <txt_label_path> -o <lmdb_label_path>
```
例如：
```bash
python tools/data/utils/txt2lmdb.py -i data/mixture/Syn90k/label.txt -o data/mixture/Syn90k/label.lmdb
```
- `TextOCR` 数据集：
  - 第一步：下载 [train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)，[TextOCR_0.1_train.json](https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json) 和 [TextOCR_0.1_val.json](https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json) 到 `textocr/` 目录.
  ```bash
  mkdir textocr && cd textocr

  # 下载 TextOCR 数据集
  wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
  wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json
  wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json

  # 对于数据图像
  unzip -q train_val_images.zip
  mv train_images train
  ```
  - 第二步：用四个并行进程剪裁图像然后生成  `train_label.txt`，`val_label.txt` ，可以使用以下命令：
  ```bash
  python tools/data/textrecog/textocr_converter.py /path/to/textocr 4
  ```


- `Totaltext` 数据集：
  - 第一步：从 [github dataset](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) 下载 `totaltext.zip`，然后从 [github Groundtruth](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Groundtruth/Text) 下载 `groundtruth_text.zip` （我们建议下载 `.mat` 格式的标注文件，因为我们提供的 `totaltext_converter.py` 标注格式转换工具只支持 `.mat` 文件）
  ```bash
  mkdir totaltext && cd totaltext
  mkdir imgs && mkdir annotations

  # 对于图像数据
  # 在 ./totaltext 目录下运行
  unzip totaltext.zip
  mv Images/Train imgs/training
  mv Images/Test imgs/test

  # 对于标注文件
  unzip groundtruth_text.zip
  cd Groundtruth
  mv Polygon/Train ../annotations/training
  mv Polygon/Test ../annotations/test

  ```
  - 第二步：用以下命令生成经剪裁后的标注文件 `train_label.txt` 和 `test_label.txt` （剪裁后的图像会被保存在目录 `data/totaltext/dst_imgs/`）：
  ```bash
  python tools/data/textrecog/totaltext_converter.py /path/to/totaltext -o /path/to/totaltext --split-list training test
  ```

## 关键信息提取

关键信息提取任务的数据集，文件目录应按如下配置：

```text
└── wildreceipt
  ├── class_list.txt
  ├── dict.txt
  ├── image_files
  ├── test.txt
  └── train.txt
```

- 下载 [wildreceipt.tar](https://download.openmmlab.com/mmocr/data/wildreceipt.tar)


## 命名实体识别（专名识别）

命名实体识别任务的数据集，文件目录应按如下配置：

```text
└── cluener2020
  ├── cluener_predict.json
  ├── dev.json
  ├── README.md
  ├── test.json
  ├── train.json
  └── vocab.txt

```

- 下载 [cluener_public.zip](https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip)

- 下载 [vocab.txt](https://download.openmmlab.com/mmocr/data/cluener_public/vocab.txt) 然后将 `vocab.txt` 移动到 `cluener2020` 文件夹下
