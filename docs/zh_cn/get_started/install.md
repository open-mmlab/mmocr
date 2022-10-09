# 安装

## 环境依赖

- Linux | Windows | macOS
- Python 3.7
- PyTorch 1.6 或更高版本
- torchvision 0.7.0
- CUDA 10.1
- NCCL 2
- GCC 5.4.0 或更高版本

## 准备环境

```{note}
如果你已经在本地安装了 PyTorch，请直接跳转到[安装步骤](#安装步骤)。
```

**第一步** 下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

**第二步** 创建并激活一个 conda 环境：

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**第三步** 依照[官方指南](https://pytorch.org/get-started/locally/)，安装 PyTorch。

在 GPU 平台上：

```shell
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## 安装步骤

我们建议大多数用户采用我们的推荐方式安装 MMOCR。倘若你需要更灵活的安装过程，则可以参考[自定义安装](#自定义安装)一节。

### 推荐步骤

**第一步** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv).

```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
```

**第二步** 将 [MMDetection](https://github.com/open-mmlab/mmdetection) 以依赖库的形式安装。

```shell
pip install 'mmdet>=3.0.0rc0'
```

**第三步** 安装 MMOCR.

情况1: 若你需要直接运行 MMOCR 或在其基础上进行开发，则通过源码安装：

```shell
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
git checkout 1.x
pip install -r requirements.txt
pip install -v -e .
# "-v" 会让安装过程产生更详细的输出
# "-e" 会以可编辑的方式安装该代码库，你对该代码库所作的任何更改都会立即生效
```

情况2：如果你将 MMOCR 作为一个外置依赖库使用，通过 pip 安装即可：

```shell
pip install 'mmocr>=1.0.0rc0'
```

**第四步（可选）** 如果你需要使用与 `albumentations` 有关的变换，比如 ABINet 数据流水线中的 `Albu`，请使用以下命令安装依赖：

```shell
# 若 MMOCR 通过源码安装
pip install -r requirements/albu.txt
# 若 MMOCR 通过 pip 安装
pip install albumentations>=1.1.0 --no-binary qudida,albumentations
```

```{note}

我们建议在安装 `albumentations` 之后检查当前环境，确保 `opencv-python` 和 `opencv-python-headless` 没有同时被安装，否则有可能会产生一些无法预知的错误。如果它们不巧同时存在于环境当中，请卸载 `opencv-python-headless` 以确保 MMOCR 的可视化工具可以正常运行。

查看 [`albumentations` 的官方文档](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies)以获知详情。

```

### 检验

根据安装方式的不同，我们提供了验证安装正确性的方法。若 MMOCR 的安装无误，你在这一节完成后应当能看到以图片和文字形式表示的识别结果，示意如下：

<div align="center">
    <img src="https://user-images.githubusercontent.com/24622904/187825445-d30cbfa6-5549-4358-97fe-245f08f4ed94.jpg" height="250"/>
</div>

```bash
# 识别结果
{'rec_texts': ['cbanke', 'docece', 'sroumats', 'chounsonse', 'doceca', 'c', '', 'sond', 'abrandso', 'sretane', '1', 'tosl', 'roundi', 'slen', 'yet', 'ally', 's', 'sue', 'salle', 'v'], 'rec_scores': [...], 'det_polygons': [...], 'det_scores': tensor([...])}
```

在 MMOCR 的目录运行以下命令：

```bash
python mmocr/ocr.py --det DB_r18 --recog CRNN demo/demo_text_ocr.jpg --show
```

也可以在 Python 解释器中运行以下代码：

```python
from mmocr.utils.ocr import MMOCR
ocr = MMOCR(recog='CRNN', det='DB_r18')
ocr.readtext('demo_text_ocr.jpg', show=True)
```

## 自定义安装

### CUDA 版本

安装 PyTorch 时，需要指定 CUDA 版本。如果您不清楚选择哪个，请遵循我们的建议：

- 对于 Ampere 架构的 NVIDIA GPU，例如 GeForce 30 series 以及 NVIDIA A100，CUDA 11 是必需的。
- 对于更早的 NVIDIA GPU，CUDA 11 是向前兼容的，但 CUDA 10.2 能够提供更好的兼容性，也更加轻量。

请确保你的 GPU 驱动版本满足最低的版本需求，参阅[这张表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```{note}
如果按照我们的最佳实践进行安装，CUDA 运行时库就足够了，因为我们提供相关 CUDA 代码的预编译，你不需要进行本地编译。
但如果你希望从源码进行 MMCV 的编译，或是进行其他 CUDA 算子的开发，那么就必须安装完整的 CUDA 工具链，参见
[NVIDIA 官网](https://developer.nvidia.com/cuda-downloads)，另外还需要确保该 CUDA 工具链的版本与 PyTorch 安装时
的配置相匹配（如用 `conda install` 安装 PyTorch 时指定的 cudatoolkit 版本）。
```

### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此其对 PyTorch 的依赖比较复杂。MIM 会自动解析这些
依赖，选择合适的 MMCV 预编译包，使安装更简单，但它并不是必需的。

要使用 pip 而不是 MIM 来安装 MMCV，请遵照 [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html)。
它需要你用指定 url 的形式手动指定对应的 PyTorch 和 CUDA 版本。

举个例子，如下命令将会安装基于 PyTorch 1.10.x 和 CUDA 11.3 编译的 mmcv-full。

```shell
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### 在 CPU 环境中安装

MMOCR 可以仅在 CPU 环境中安装，在 CPU 模式下，你可以完成训练（需要 MMCV 版本 >= 1.4.4）、测试和模型推理等所有操作。

在 CPU 模式下，MMCV 中的以下算子将不可用：

- Deformable Convolution
- Modulated Deformable Convolution
- ROI pooling
- SyncBatchNorm

如果你尝试使用用到了以上算子的模型进行训练、测试或推理，程序将会报错。以下为可能受到影响的模型列表：

|                          算子                           |                          模型                           |
| :-----------------------------------------------------: | :-----------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DBNet (r50dcnv2), DBNet++ (r50dcnv2), FCENet (r50dcnv2) |
|                      SyncBatchNorm                      |                      PANet, PSENet                      |

### 通过 Docker 使用 MMOCR

我们提供了一个 [Dockerfile](https://github.com/open-mmlab/mmocr/blob/master/docker/Dockerfile) 文件以建立 docker 镜像 。

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmocr docker/
```

使用以下命令运行。

```shell
docker run --gpus all --shm-size=8g -it -v {实际数据目录}:/mmocr/data mmocr
```

## 对 MMCV 和 MMDetection 的版本依赖

为了确保代码实现的正确性，MMOCR 每个版本都有可能改变对 MMCV 和 MMDetection 版本的依赖。请根据以下表格确保版本之间的相互匹配。

| MMOCR         | MMCV              | MMDetection        |
| ------------- | ----------------- | ------------------ |
| dev-1.x       | 2.0.0rc1 \<= mmcv | 3.0.0rc0 \<= mmdet |
| 1.0.0rc0, rc1 | 2.0.0rc1 \<= mmcv | 3.0.0rc0 \<= mmdet |
