
# 文字检测

## 概览

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
```

|  数据集名称  |                             数据图片                             |                                                                                                     |               标注文件                  |                                                                                                |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :-------------------------------------: | :--------------------------------------------------------------------------------------------: |
|           |                                                                                      |                                                训练集 (training)                                                |               验证集 (validation)                |                                           测试集 (testing)                                             |       |
|  CTW1500  | [下载地址](https://github.com/Yuliang-Liu/Curve-Text-Detector) |                    -                    |                    -                    |                    -                    |
| ICDAR2015 | [下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads)     | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json) |                    -                    | [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json) |
| ICDAR2017 | [下载地址](https://rrc.cvc.uab.es/?ch=8&com=downloads)     | [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_training.json) | [instances_val.json](https://download.openmmlab.com/mmocr/data/icdar2017/instances_val.json) | - |       |       |
| Synthtext | [下载地址](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)  | instances_training.lmdb ([data.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/data.mdb), [lock.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/lock.mdb)) |                    -                    | - |
| TextOCR | [下载地址](https://textvqa.org/textocr/dataset)  | - |                    -                    | -
| Totaltext | [下载地址](https://github.com/cs-chan/Total-Text-Dataset)  | - |                    -                    | -

## 重要提醒

:::{note}
**若用户需要在 CTW1500, ICDAR 2015/2017 或 Totaltext 数据集上训练模型**, 请注意这些数据集中有部分图片的 EXIF 信息里保存着方向信息。MMCV 采用的 OpenCV 后端会默认根据方向信息对图片进行旋转；而由于数据集的标注是在原图片上进行的，这种冲突会使得部分训练样本失效。因此，用户应该在配置 pipeline 时使用 `dict(type='LoadImageFromFile', color_type='color_ignore_orientation')` 以避免 MMCV 的这一行为。（配置文件可参考 [DBNet 的 pipeline 配置](https://github.com/open-mmlab/mmocr/blob/main/configs/_base_/det_pipelines/dbnet_pipeline.py)）
:::


## 准备步骤

### ICDAR 2015
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

### ICDAR 2017
- 与上述步骤类似。

### CTW1500
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

### SynthText

- 下载 [data.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/data.mdb) 和 [lock.mdb](https://download.openmmlab.com/mmocr/data/synthtext/instances_training.lmdb/lock.mdb) 并放置到 `synthtext/instances_training.lmdb/` 中.

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

### Totaltext
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
