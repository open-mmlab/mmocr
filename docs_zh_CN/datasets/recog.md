# 文字识别

## 概览

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
|  MJSynth (Syn90k) |               [下载地址](https://www.robots.ox.ac.uk/~vgg/data/text/)                |                                                       [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/shuffle_labels.txt) \| [label.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/label.txt)                                                       |                                                    -                                                    |       |
| SynthText (Synth800k) |           [下载地址](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)              | [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/shuffle_labels.txt) \| [instances_train.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/instances_train.txt) \| [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthText/label.txt) |                                                    -                                                    |       |
|  SynthAdd  |  [SynthText_Add.zip](https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg)  (code:627x)   |                                                                                                           [label.txt](https://download.openmmlab.com/mmocr/data/mixture/SynthAdd/label.txt)                                                                                                            |                                                    -                                                    |       |
|  TextOCR  |  [下载地址](https://textvqa.org/textocr/dataset)   |                                                                                                           -                                                                                                           |                                                    -                                                    |       |
|  Totaltext  |  [下载地址](https://github.com/cs-chan/Total-Text-Dataset)   |                                                                                                           -                                                                                                           |                                                    -                                                    |       |

(*) 注：由于官方的下载地址已经无法访问，我们提供了一个非官方的地址以供参考，但我们无法保证数据的准确性。

## 准备步骤

### ICDAR 2013
- 第一步：从 [下载地址](https://rrc.cvc.uab.es/?ch=2&com=downloads) 下载 `Challenge2_Test_Task3_Images.zip` 和 `Challenge2_Training_Task3_Images_GT.zip`
- 第二步：下载 [test_label_1015.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/test_label_1015.txt) 和 [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2013/train_label.txt)

### ICDAR 2015
- 第一步：从 [下载地址](https://rrc.cvc.uab.es/?ch=4&com=downloads) 下载 `ch4_training_word_images_gt.zip` 和 `ch4_test_word_images_gt.zip`
- 第二步：下载 [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt) and [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt)

### IIIT5K
- 第一步：从 [下载地址](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) 下载 `IIIT5K-Word_V3.0.tar.gz`
- 第二步：下载 [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/train_label.txt) 和 [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)

### svt
- 第一步：从 [下载地址](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) 下载 `svt.zip`
- 第二步：下载 [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svt/test_label.txt)
- 第三步：
```bash
python tools/data/textrecog/svt_converter.py <download_svt_dir_path>
```

### ct80
- 第一步：下载 [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/ct80/test_label.txt)

### svtp
- 第一步：下载 [test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/svtp/test_label.txt)

### coco_text
  - 第一步：从 [下载地址](https://rrc.cvc.uab.es/?ch=5&com=downloads) 下载文件
  - 第二步：下载 [train_label.txt](https://download.openmmlab.com/mmocr/data/mixture/coco_text/train_label.txt)

### MJSynth (Syn90k)
  - 第一步：从 [下载地址](https://www.robots.ox.ac.uk/~vgg/data/text/) 下载 `mjsynth.tar.gz`
  - 第二步：下载 [shuffle_labels.txt](https://download.openmmlab.com/mmocr/data/mixture/Syn90k/shuffle_labels.txt)
  - 第三步：

  ```bash
  mkdir Syn90k && cd Syn90k

  mv /path/to/mjsynth.tar.gz .

  tar -xzf mjsynth.tar.gz

  mv /path/to/shuffle_labels.txt .
  mv /path/to/label.txt .

  # 创建软链接
  cd /path/to/mmocr/data/mixture

  ln -s /path/to/Syn90k Syn90k
  ```

### SynthText (Synth800k)
  - 第一步： 从 [下载地址](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) 下载 `SynthText.zip`
  - 第二步：

  ```bash
  mkdir SynthText && cd SynthText
  mv /path/to/SynthText.zip .
  unzip SynthText.zip
  mv SynthText synthtext

  mv /path/to/shuffle_labels.txt .
mv /path/to/label.txt .

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

### SynthAdd
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
:::{tips}
运行以下命令，可以把 `.txt` 格式的标注文件转换成 `.lmdb` 格式：
```bash
python tools/data/utils/txt2lmdb.py -i <txt_label_path> -o <lmdb_label_path>
```
例如：
```bash
python tools/data/utils/txt2lmdb.py -i data/mixture/Syn90k/label.txt -o data/mixture/Syn90k/label.lmdb
```
:::

### TextOCR
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


### Totaltext
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
