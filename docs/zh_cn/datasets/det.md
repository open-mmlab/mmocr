
# 文字检测

## 概览



|  数据集名称  |                             数据图片                             |                                                                                                     |               标注文件                  |                                                                                                |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :-------------------------------------: | :--------------------------------------------------------------------------------------------: |
|           |                                                                                      |                                                训练集 (training)                                                |               验证集 (validation)                |                                           测试集 (testing)                                             |       |
|  CTW1500  | [下载地址](https://github.com/Yuliang-Liu/Curve-Text-Detector) |                    -                    |                    -                    |                    -                    |
| ICDAR2011 | [下载地址](https://rrc.cvc.uab.es/?ch=1)     |                    -                    |                    -                    |                    -                    |
| ICDAR2013 | [下载地址](https://rrc.cvc.uab.es/?ch=2)     |                    -                    |                    -                    |                    -                    |
| ICDAR2015 | [下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads)     | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json) |                    -                    | [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json) |
| ICDAR2017 | [下载地址](https://rrc.cvc.uab.es/?ch=8&com=downloads)     | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_training.json) | [instances_val.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_val.json) | - |       |       |
| Synthtext | [下载地址](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)  | instances_training.lmdb ([data.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/data.mdb), [lock.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/lock.mdb)) |                    -                    | - |
| TextOCR | [下载地址](https://textvqa.org/textocr/dataset)  | - |                    -                    | -
| Totaltext | [下载地址](https://github.com/cs-chan/Total-Text-Dataset)  | - |                    -                    | -
| CurvedSynText150k | [下载地址](https://github.com/aim-uofa/AdelaiDet/blob/master/datasets/README.md) \| [Part1](https://drive.google.com/file/d/1OSJ-zId2h3t_-I7g_wUkrK-VqQy153Kj/view?usp=sharing) \| [Part2](https://drive.google.com/file/d/1EzkcOlIgEp5wmEubvHb7-J5EImHExYgY/view?usp=sharing) |                                                          [instances_training.json](https://download.openmmlab.com/mmocr/data/curvedsyntext/instances_training.json)                                                          |                                              -                                               |   -
|FUNSD|[下载地址](https://guillaumejaume.github.io/FUNSD/)|-|-|-|  
|DeText|[下载地址](https://rrc.cvc.uab.es/?ch=9)|-|-|-|
|NAF|[下载地址](https://github.com/herobd/NAF_dataset/releases/tag/v1.0)|-|-|-|                                                                         
|SROIE|[下载地址](https://rrc.cvc.uab.es/?ch=13)|-|-|-|       
|Lecture Video DB|[下载地址](https://cvit.iiit.ac.in/research/projects/cvit-projects/lecturevideodb)|-|-|-|                                                                         
|LSVT|[下载地址](https://rrc.cvc.uab.es/？ch=16)|-|-|-|                                                                         
|IMGUR|[下载地址](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset)|-|-|-|
|KAIST|[下载地址](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database)|-|-|-|                                |
|MTWI|[下载地址](https://tianchi.aliyun.com/competition/entrance/231685/information?lang=en-us) |-|-|-|                                                            
|COCO Text v2|[下载地址](https://bgshih.github.io/cocotext/)|-|-|-| 
|ReCTS|[下载地址](https://rrc.cvc.uab.es/?ch=12)|-|-|-|
|IIIT-ILST|[下载地址](http://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-ilst)|-|-|-|                                                                 
|VinText|[下载地址](https://github.com/VinAIResearch/dict-guided)|-|-|-|
|BID|[下载地址](https://github.com/ricardobnjunior/Brazilian-Identity-Document-Dataset)|-|-|-|
|RCTW|[下载地址](https://rctw.vlrlab.net/index.html)|-|-|-| 

## 重要提醒

:::{note}
**若用户需要在 CTW1500, ICDAR 2015/2017 或 Totaltext 数据集上训练模型**, 请注意这些数据集中有部分图片的 EXIF 信息里保存着方向信息。MMCV 采用的 OpenCV 后端会默认根据方向信息对图片进行旋转；而由于数据集的标注是在原图片上进行的，这种冲突会使得部分训练样本失效。因此，用户应该在配置 pipeline 时使用 `dict(type='LoadImageFromFile', color_type='color_ignore_orientation')` 以避免 MMCV 的这一行为。（配置文件可参考 [DBNet 的 pipeline 配置](https://github.com/open-mmlab/mmocr/blob/main/configs/_base_/det_pipelines/dbnet_pipeline.py)）
:::


## 准备步骤

## CTW1500

- 第一步: 运行以下命令, 从 [github](https://github.com/Yuliang-Liu/Curve-Text-Detector)下载`train_images.zip`, `test_images.zip`, `train_labels.zip`, `test_labels.zip`等四个文件。

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

- 第二步: 运行以下命令，生成 `instances_training.json` 和 `instances_test.json`:

  ```bash
  python tools/data/textdet/ctw1500_converter.py /path/to/ctw1500 -o /path/to/ctw1500 --split-list training test
  ```

- 所产生的目录配置应如下:

  ```text
  ├── ctw1500
  │   ├── imgs
  │   ├── annotations
  │   ├── instances_training.json
  │   └── instances_val.json
  ```
## ICDAR 2011 (原生数位图像 Born-Digital Images)

- 第一步: 运行以下命令, 从[下载地址](https://rrc.cvc.uab.es/?ch=1&com=downloads) `Task 1.1: Text Localization (2013 edition)`下载`Challenge1_Training_Task12_Images.zip`, `Challenge1_Training_Task1_GT.zip`, `Challenge1_Test_Task12_Images.zip`, 和 `Challenge1_Test_Task1_GT.zip`。

  ```bash
  mkdir icdar2011 && cd icdar2011
  mkdir imgs && mkdir annotations

  # 下载 ICDAR 2011
  wget https://rrc.cvc.uab.es/downloads/Challenge1_Training_Task12_Images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge1_Training_Task1_GT.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge1_Test_Task12_Images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge1_Test_Task1_GT.zip --no-check-certificate

  # 下载并配置数据
  unzip -q Challenge1_Training_Task12_Images.zip -d imgs/training
  unzip -q Challenge1_Test_Task12_Images.zip -d imgs/test
  # 下载并配置标注
  unzip -q Challenge1_Training_Task1_GT.zip -d annotations/training
  unzip -q Challenge1_Test_Task1_GT.zip -d annotations/test

  rm Challenge1_Training_Task12_Images.zip && rm Challenge1_Test_Task12_Images.zip && rm Challenge1_Training_Task1_GT.zip && rm Challenge1_Test_Task1_GT.zip
  ```

- 第二步: 运行以下命令，生成 `instances_training.json` 和 `instances_test.json`:

  ```bash
  python tools/data/textdet/ic11_converter.py PATH/TO/icdar2011 --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── icdar2011
  │   ├── imgs
  │   ├── instances_test.json
  │   └── instances_training.json
  ```

## ICDAR 2013 (聚焦场景文本 Focused Scene Text)

- 第一步: 运行以下命令, 从[下载地址](https://rrc.cvc.uab.es/?ch=2&com=downloads)`Task 2.1: Text Localization (2013 edition)` 下载`Challenge2_Training_Task12_Images.zip`, `Challenge2_Test_Task12_Images.zip`, `Challenge2_Training_Task1_GT.zip`, 和 `Challenge2_Test_Task1_GT.zip`。

  ```bash
  mkdir icdar2013 && cd icdar2013
  mkdir imgs && mkdir annotations

  # 下载 ICDAR 2013
  wget https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task12_Images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task12_Images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task1_GT.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task1_GT.zip --no-check-certificate

  # 下载并配置数据
  unzip -q Challenge2_Training_Task12_Images.zip -d imgs/training
  unzip -q Challenge2_Test_Task12_Images.zip -d imgs/test
  # 下载并配置标注
  unzip -q Challenge2_Training_Task1_GT.zip -d annotations/training
  unzip -q Challenge2_Test_Task1_GT.zip -d annotations/test

  rm Challenge2_Training_Task12_Images.zip && rm Challenge2_Test_Task12_Images.zip && rm Challenge2_Training_Task1_GT.zip && rm Challenge2_Test_Task1_GT.zip
  ```

- 第二步: 运行以下命令，生成 `instances_training.json` 和 `instances_test.json`:

  ```bash
  python tools/data/textdet/ic13_converter.py PATH/TO/icdar2013 --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── icdar2013
  │   ├── imgs
  │   ├── instances_test.json
  │   └── instances_training.json
  ```

### ICDAR 2015
- 第一步：从[下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载 `ch4_training_images.zip`、`ch4_test_images.zip`、`ch4_training_localization_transcription_gt.zip`、`Challenge4_Test_Task1_GT.zip` 四个文件，分别对应训练集数据、测试集数据、训练集标注、测试集标注。
- 第二步：运行以下命令，移动数据集到对应文件夹
```bash
mkdir icdar2015 && cd icdar2015
mkdir imgs && mkdir annotations
# 移动图片到目录：
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

- 所产生的目录配置应如下:

  ```text
  ├── icdar2015
  │   ├── imgs
  │   ├── annotations
  │   ├── instances_test.json
  │   └── instances_training.json
  ```

### ICDAR 2017
- 与上述步骤类似。
- 所产生的目录配置应如下:

  ```text
  ├── icdar2017
  │   ├── imgs
  │   ├── annotations
  │   ├── instances_training.json
  │   └── instances_val.json
  ```


### SynthText

- 第一步：从 [下载地址](<https://www.robots.ox.ac.uk/~vgg/data/scenetext/>) 下载 Download SynthText.zip 然后解压缩到`synthtext/img`.

- 第二步: 下载 [data.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/data.mdb) 和 [lock.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/lock.mdb) 并放置到 `synthtext/instances_training.lmdb/`中.

- 所产生的目录配置应如下:

  ```text
  ├── synthtext
  │   ├── imgs
  │   └── instances_training.lmdb
  │       ├── data.mdb
  │       └── lock.mdb
  ```

### TextOCR
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
  - 运行以上命令后, 所产生的目录配置应如下:

  ```text
  ├── textocr
  │   ├── train
  │   ├── instances_training.json
  │   └── instances_val.json
  ```


### Totaltext
  - 第一步：从 [github dataset](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) 下载 `totaltext.zip`，从 [github Groundtruth](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Groundtruth/Text) 下载 `groundtruth_text.zip` 。（我们提供的标注格式转换脚本 `totaltext_converter.py` 支持 `.mat` 和 `.text.` 格式。）
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

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  ├── totaltext
  │   ├── imgs
  │   ├── annotations
  │   ├── instances_test.json
  │   └── instances_training.json
  ```

## CurvedSynText150k

- 第一步: 下载 [syntext1.zip](https://drive.google.com/file/d/1OSJ-zId2h3t_-I7g_wUkrK-VqQy153Kj/view?usp=sharing) 和 [syntext2.zip](https://drive.google.com/file/d/1EzkcOlIgEp5wmEubvHb7-J5EImHExYgY/view?usp=sharing) 两个文件到 `CurvedSynText150k/`中.

- 第二步:

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

- 第三步: 下载 [instances_training.json](https://download.openmmlab.com/mmocr/data/curvedsyntext/instances_training.json) 到 `CurvedSynText150k/`
- 或是用以下命令直接生成 `instances_training.json` :

  ```bash
  python tools/data/common/curvedsyntext_converter.py PATH/TO/CurvedSynText150k --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  ├── CurvedSynText150k
  │   ├── syntext_word_eng
  │   ├── emcs_imgs
  │   └── instances_training.json
  ```

## FUNSD

- 第一步: 下载 [dataset.zip](https://guillaumejaume.github.io/FUNSD/dataset.zip) 到 `funsd/`中。 

  ```bash
  mkdir funsd && cd funsd

  # 下载 FUNSD 数据集
  wget https://guillaumejaume.github.io/FUNSD/dataset.zip
  unzip -q dataset.zip

  # 移动图片到目录
  mv dataset/training_data/images imgs && mv dataset/testing_data/images/* imgs/

  # 移动标注到目录
  mkdir annotations
  mv dataset/training_data/annotations annotations/training && mv dataset/testing_data/annotations annotations/test

  rm dataset.zip && rm -rf dataset
  ```

- 第二步: 使用以下命令，生成 `instances_training.json` 和 `instances_test.json`：

  ```bash
  python tools/data/textdet/funsd_converter.py PATH/TO/funsd --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── funsd
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_test.json
  │   └── instances_training.json
  ```

## DeText

- 第一步: 从[homepage](https://rrc.cvc.uab.es/?ch=9)**Task 3: End to End**下载 `ch9_training_images.zip`, `ch9_training_localization_transcription_gt.zip`, `ch9_validation_images.zip`, 和 `ch9_validation_localization_transcription_gt.zip`。 

  ```bash
  mkdir detext && cd detext
  mkdir imgs && mkdir annotations && mkdir imgs/training && mkdir imgs/val && mkdir annotations/training && mkdir annotations/val

  # 下载 DeText
  wget https://rrc.cvc.uab.es/downloads/ch9_training_images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch9_training_localization_transcription_gt.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch9_validation_images.zip --no-check-certificate
  wget https://rrc.cvc.uab.es/downloads/ch9_validation_localization_transcription_gt.zip --no-check-certificate

  # 提取图片和标注
  unzip -q ch9_training_images.zip -d imgs/training && unzip -q ch9_training_localization_transcription_gt.zip -d annotations/training && unzip -q ch9_validation_images.zip -d imgs/val && unzip -q ch9_validation_localization_transcription_gt.zip -d annotations/val

  # 移除压缩档
  rm ch9_training_images.zip && rm ch9_training_localization_transcription_gt.zip && rm ch9_validation_images.zip && rm ch9_validation_localization_transcription_gt.zip
  ```

- 第二步: 使用以下命令，生成 `instances_training.json` 和 `instances_val.json`:

  ```bash
  python tools/data/textdet/detext_converter.py PATH/TO/detext --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── detext
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_test.json
  │   └── instances_training.json
  ```

## NAF

- 第一步: 下载 [labeled_images.tar.gz](https://github.com/herobd/NAF_dataset/releases/tag/v1.0) 到 `naf/`中。

  ```bash
  mkdir naf && cd naf

  # 下载 NAF 数据集
  wget https://github.com/herobd/NAF_dataset/releases/download/v1.0/labeled_images.tar.gz
  tar -zxf labeled_images.tar.gz

  # 移动图片到目录
  mkdir annotations && mv labeled_images imgs

  # 移动标注到目录
  git clone https://github.com/herobd/NAF_dataset.git
  mv NAF_dataset/train_valid_test_split.json annotations/ && mv NAF_dataset/groups annotations/

  rm -rf NAF_dataset && rm labeled_images.tar.gz
  ```

- 第二步: 用以下命令，生成 `instances_training.json`, `instances_val.json`, 和 `instances_test.json` :

  ```bash
  python tools/data/textdet/naf_converter.py PATH/TO/naf --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── naf
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_test.json
  │   ├── instances_val.json
  │   └── instances_training.json
  ```

## SROIE

- 第一步: 从[下载地址](https://rrc.cvc.uab.es/?ch=13&com=downloads) 下载 `0325updated.task1train(626p).zip`, `task1&2_test(361p).zip`, 和 `text.task1&2-test（361p).zip` 到 `sroie/`中。

- 第二步:

  ```bash
  mkdir sroie && cd sroie
  mkdir imgs && mkdir annotations && mkdir imgs/training

  # 警告: 从 Google Drive 和 百度云网盘所下载的压缩档档案名可能不同，如果用户在提取或移动档案时有出现错误讯息，请将以下命令中的档案名改成正确应对的档案名。
  unzip -q 0325updated.task1train\(626p\).zip && unzip -q task1\&2_test\(361p\).zip && unzip -q text.task1\&2-test（361p\).zip

  # 移动图片到目录
  mv 0325updated.task1train\(626p\)/*.jpg imgs/training && mv fulltext_test\(361p\) imgs/test

  # 移动标注到目录
  mv 0325updated.task1train\(626p\) annotations/training && mv text.task1\&2-testги361p\)/ annotations/test

  rm 0325updated.task1train\(626p\).zip && rm task1\&2_test\(361p\).zip && rm text.task1\&2-test（361p\).zip
  ```

- 第三步: 使用以下命令， 生成 `instances_training.json` 和 `instances_test.json` :

  ```bash
  python tools/data/textdet/sroie_converter.py PATH/TO/sroie --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  ├── sroie
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_test.json
  │   └── instances_training.json
  ```

## Lecture Video DB

- 第一步: 下载 [IIIT-CVid.zip](http://cdn.iiit.ac.in/cdn/preon.iiit.ac.in/~kartik/IIIT-CVid.zip) 到 `lv/`中。

  ```bash
  mkdir lv && cd lv

  # 下载 LV 数据集
  wget http://cdn.iiit.ac.in/cdn/preon.iiit.ac.in/~kartik/IIIT-CVid.zip
  unzip -q IIIT-CVid.zip

  mv IIIT-CVid/Frames imgs

  rm IIIT-CVid.zip
  ```

- 第二步: 使用以下命令， 生成 `instances_training.json`, `instances_val.json`, 和 `instances_test.json`：
  ```bash
  python tools/data/textdet/lv_converter.py PATH/TO/lv --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── lv
  │   ├── imgs
  │   ├── instances_test.json
  │   ├── instances_training.json
  │   └── instances_val.json
  ```

### LSVT

- 第一步: 下载 [train_full_images_0.tar.gz](https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz), [train_full_images_1.tar.gz](https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz), 和 [train_full_labels.json](https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json) 到 `lsvt/`中。

  ```bash
  mkdir lsvt && cd lsvt

  # 下载 LSVT 数据集
  wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz
  wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz
  wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json

  mkdir annotations
  tar -xf train_full_images_0.tar.gz && tar -xf train_full_images_1.tar.gz
  mv train_full_labels.json annotations/ && mv train_full_images_1/*.jpg train_full_images_0/
  mv train_full_images_0 imgs

  rm train_full_images_0.tar.gz && rm train_full_images_1.tar.gz && rm -rf train_full_images_1
  ```

- 第二步: 使用以下命令， 生成 `instances_training.json` 和 `instances_val.json` (自选)：

  ```bash
  # LSVT没有公开拆分测试集的标注所以我们拆分验证集
  # 加 --val-ratio 0.2
  python tools/data/textdet/lsvt_converter.py PATH/TO/lsvt
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  |── lsvt
  │   ├── imgs
  │   ├── instances_training.json
  │   └── instances_val.json (optional)
  ```

## IMGUR

- 第一步: 运行 `download_imgur5k.py` 下载图片。 用户可以合并 [PR#5](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset/pull/5) 到自己的本地仓库，从以达到**更快** 的并行处理图片下载速度。

  ```bash
  mkdir imgur && cd imgur

  git clone https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset.git

  # 从imgur.com下载图片。 可能会需要几个小时！
  python ./IMGUR5K-Handwriting-Dataset/download_imgur5k.py --dataset_info_dir ./IMGUR5K-Handwriting-Dataset/dataset_info/ --output_dir ./imgs

  # 移动标注到目录
  mkdir annotations
  mv ./IMGUR5K-Handwriting-Dataset/dataset_info/*.json annotations

  rm -rf IMGUR5K-Handwriting-Dataset
  ```

- 第二步: 使用以下命令，生成 `instances_train.json`, `instance_val.json` 和 `instances_test.json` ：

  ```bash
  python tools/data/textdet/imgur_converter.py PATH/TO/imgur
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```
  │── imgur
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_test.json
  │   ├── instances_training.json
  │   └── instances_val.json
  ```

## KAIST

- 第一步: 下载 [KAIST_all.zip](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database) 到 `kaist/`中。

  ```bash
  mkdir kaist && cd kaist
  mkdir imgs && mkdir annotations

  # 下载 KAIST 数据集
  wget http://www.iapr-tc11.org/dataset/KAIST_SceneText/KAIST_all.zip
  unzip -q KAIST_all.zip

  rm KAIST_all.zip
  ```

- 第二步: 提取压缩档:

  ```bash
  python tools/data/common/extract_kaist.py PATH/TO/kaist
  ```

- 第三步: 使用以下命令， 生成 `instances_training.json` 和 `instances_val.json` (自选)：

  ```bash
  # 因为 KASIT 没有提供正式的拆分所以用户可以加--val-ratio 0.2来拆分数据集
  python tools/data/textdet/kaist_converter.py PATH/TO/kaist --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── kaist
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_training.json
  │   └── instances_val.json (optional)
  ```

## MTWI

- 第一步: 从[下载地址](https://tianchi.aliyun.com/competition/entrance/231685/information?lang=en-us)下载 `mtwi_2018_train.zip`.

  ```bash
  mkdir mtwi && cd mtwi

  unzip -q mtwi_2018_train.zip
  mv image_train imgs && mv txt_train annotations

  rm mtwi_2018_train.zip
  ```

- 第二步: 使用以下命令，生成 `instances_training.json` and `instance_val.json` (自选)：
  ```bash
  # MTWI没有公开拆分测试集的标注所以我们拆分验证集
  # 加 --val-ratio 0.2
  python tools/data/textdet/mtwi_converter.py PATH/TO/mtwi --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── mtwi
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_training.json
  │   └── instances_val.json (optional)
  ```

## COCO Text v2

- 第一步: 下载图片 [train2014.zip](http://images.cocodataset.org/zips/train2014.zip) 和 标注 [cocotext.v2.zip](https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip) 到 `coco_textv2/`中。

  ```bash
  mkdir coco_textv2 && cd coco_textv2
  mkdir annotations

  # 下载 COCO Text v2 数据集
  wget http://images.cocodataset.org/zips/train2014.zip
  wget https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip
  unzip -q train2014.zip && unzip -q cocotext.v2.zip

  mv train2014 imgs && mv cocotext.v2.json annotations/

  rm train2014.zip && rm -rf cocotext.v2.zip
  ```

- 第二步: 使用以下命令，生成 `instances_training.json` 和 `instances_val.json` ：

  ```bash
  python tools/data/textdet/cocotext_converter.py PATH/TO/coco_textv2
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── coco_textv2
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_training.json
  │   └── instances_val.json
  ```

## ReCTS

- 第一步: 从[homepage](https://rrc.cvc.uab.es/?ch=12&com=downloads) 下载 [ReCTS.zip](https://datasets.cvc.uab.es/rrc/ReCTS.zip) 到 `rects/` 中。

  ```bash
  mkdir rects && cd rects

  # 下载 ReCTS 数据集
  # 用户可以在数据集的主页上找到Google Drive的链接
  wget https://datasets.cvc.uab.es/rrc/ReCTS.zip --no-check-certificate
  unzip -q ReCTS.zip

  mv img imgs && mv gt_unicode annotations

  rm ReCTS.zip && rm -rf gt
  ```

- 第二步: 使用以下命令， 生成 `instances_training.json` 和 `instances_val.json` (自选):

  ```bash
  # ReCTS没有公开拆分测试集的标注所以我们可以加 --val-ratio 0.2来拆分验证集
  # 加 --preserve-vertical来保留纵向文字档做为训练用， 否则，纵向图像被筛选出之后会保存在 PATH/TO/rects/ignores
  python tools/data/textdet/rects_converter.py PATH/TO/rects --nproc 4 --val-ratio 0.2
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── rects
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_val.json (optional)
  │   └── instances_training.json
  ```

## ILST

- 第一步: 从 [onedrive](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/minesh_mathew_research_iiit_ac_in/EtLvCozBgaBIoqglF4M-lHABMgNcCDW9rJYKKWpeSQEElQ?e=zToXZP)下载 `IIIT-ILST` 

- 第二步: 运行以下命令

  ```bash
  unzip -q IIIT-ILST.zip && rm IIIT-ILST.zip
  cd IIIT-ILST

  # 重新命名档案
  cd Devanagari && for i in `ls`; do mv -f $i `echo "devanagari_"$i`; done && cd ..
  cd Malayalam && for i in `ls`; do mv -f $i `echo "malayalam_"$i`; done && cd ..
  cd Telugu && for i in `ls`; do mv -f $i `echo "telugu_"$i`; done && cd ..

  # 移动图像路径
  mkdir imgs && mkdir annotations
  mv Malayalam/{*jpg,*jpeg} imgs/ && mv Malayalam/*xml annotations/
  mv Devanagari/*jpg imgs/ && mv Devanagari/*xml annotations/
  mv Telugu/*jpeg imgs/ && mv Telugu/*xml annotations/

  # 移除多余的档案
  rm -rf Devanagari && rm -rf Malayalam && rm -rf Telugu && rm -rf README.txt
  ```

- 第三步: 生成 `instances_training.json` 和 `instances_val.json` (自选)。因为原本的数据集没有包含验证集， 用户可以用 `--val-ratio` 来拆分数据集。 比方说，如果设 val-ratio 为 0.2， 20% 的数据就会从原本的数据集被拆分开留作验证集。 

  ```bash
  python tools/data/textdet/ilst_converter.py    PATH/TO/IIIT-ILST --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── IIIT-ILST
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_val.json (optional)
  │   └── instances_training.json
  ```

## VinText

- 第一步: 下载 [vintext.zip](https://drive.google.com/drive/my-drive) 到 `vintext`中。

  ```bash
  mkdir vintext && cd vintext

  # 从google drive下载数据集
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml' -O- │ sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml" -O vintext.zip && rm -rf /tmp/cookies.txt

  # 提取图片和标注
  unzip -q vintext.zip && rm vintext.zip
  mv vietnamese/labels ./ && mv vietnamese/test_image ./ && mv vietnamese/train_images ./ && mv vietnamese/unseen_test_images ./
  rm -rf vietnamese

  # 重新命名档案
  mv labels annotations && mv test_image test && mv train_images  training && mv unseen_test_images  unseen_test
  mkdir imgs
  mv training imgs/ && mv test imgs/ && mv unseen_test imgs/
  ```

- 第二步: 生成 `instances_training.json`, `instances_test.json` 和 `instances_unseen_test.json`

  ```bash
  python tools/data/textdet/vintext_converter.py PATH/TO/vintext --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── vintext
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_test.json
  │   ├── instances_unseen_test.json
  │   └── instances_training.json
  ```

## BID

- 第一步: 下载 [BID Dataset.zip](https://drive.google.com/file/d/1Oi88TRcpdjZmJ79WDLb9qFlBNG8q2De6/view)

- 第二步: 运行以下命令来预处理数据集

  ```bash
  # 重新命名
  mv BID\ Dataset.zip BID_Dataset.zip

  # 解压缩和重新命名
  unzip -q BID_Dataset.zip && rm BID_Dataset.zip
  mv BID\ Dataset BID

  # BID数据集有许可问题，用户可以给予这个档案许可
  chmod -R 777 BID
  cd BID
  mkdir imgs && mkdir annotations

  # 移动图片和标注
  mv CNH_Aberta/*in.jpg imgs && mv CNH_Aberta/*txt annotations && rm -rf CNH_Aberta
  mv CNH_Frente/*in.jpg imgs && mv CNH_Frente/*txt annotations && rm -rf CNH_Frente
  mv CNH_Verso/*in.jpg imgs && mv CNH_Verso/*txt annotations && rm -rf CNH_Verso
  mv CPF_Frente/*in.jpg imgs && mv CPF_Frente/*txt annotations && rm -rf CPF_Frente
  mv CPF_Verso/*in.jpg imgs && mv CPF_Verso/*txt annotations && rm -rf CPF_Verso
  mv RG_Aberto/*in.jpg imgs && mv RG_Aberto/*txt annotations && rm -rf RG_Aberto
  mv RG_Frente/*in.jpg imgs && mv RG_Frente/*txt annotations && rm -rf RG_Frente
  mv RG_Verso/*in.jpg imgs && mv RG_Verso/*txt annotations && rm -rf RG_Verso

  # 移除非必要档案
  rm -rf desktop.ini
  ```

- 第三步: 生成 `instances_training.json` 和 `instances_val.json` (自选)。 因为原本的数据集没有包含验证集， 用户可以用 `--val-ratio` 来拆分数据集。 比方说，如果设 val-ratio 为 0.2， 20% 的数据就会从原本的数据集被拆分开留作验证集。  

  ```bash
  python tools/data/textdet/bid_converter.py PATH/TO/BID --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── BID
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_training.json
  │   └── instances_val.json (optional)
  ```

## RCTW

- 第一步: 从[下载地址](https://rctw.vlrlab.net/dataset.html)下载 `train_images.zip.001`, `train_images.zip.002`, 和 `train_gts.zip`, 然后解压缩档案 `rctw/imgs` 和 `rctw/annotations`。

- 第二步: 生成 `instances_training.json` 和 `instances_val.json` (自选)。 因为原本的数据集没有包含验证集， 用户可以用 `--val-ratio` 来拆分数据集。 比方说，如果设 val-ratio 为 0.2， 20% 的数据就会从原本的数据集被拆分开留作验证集。  

  ```bash
  # RCTW没有公开拆分测试集的标注所以我们可以加--val-ratio 0.2来拆分验证集
  # 加 --preserve-vertical， 保留纵向文字档做为训练用。否则纵向图像被筛选出之后会保存在 PATH/TO/rects/ignores
  python tools/data/textdet/rctw_converter.py PATH/TO/rctw --nproc 4
  ```

- 运行以上命令后, 所产生的目录配置应如下:

  ```text
  │── rctw
  │   ├── annotations
  │   ├── imgs
  │   ├── instances_training.json
  │   └── instances_val.json (optional)
  ```
