# Installation

## Prerequisites

- Linux (Windows is not officially supported)
- Python 3.7
- PyTorch 1.5 or higher
- torchvision 0.6.0
- CUDA 10.1
- NCCL 2
- GCC 5.4.0 or higher
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) 1.3.4
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/#installation) 2.11.0

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04
- CUDA: 10.1
- GCC(G++): 5.4.0
- MMCV 1.3.4
- MMDetection 2.11.0
- PyTorch 1.5
- torchvision 0.6.0

MMOCR depends on Pytorch and mmdetection.

## Step-by-Step Installation Instructions

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```
Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).


c. Install mmcv, we recommend you to install the pre-build mmcv as below.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired one. For example, to install the latest ``mmcv-full`` with ``CUDA 11`` and ``PyTorch 1.7.0``, use the following command:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```
Note that mmocr 0.2.0 or later requires mmcv 1.3.4 or later.

If it compiles during installation, then please check that the cuda version and pytorch version **exactly"" matches the version in the mmcv-full installation command. For example, pytorch 1.7.0 and 1.7.1 are treated differently.

See official [installation](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

**Important:** You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

d. Install [mmdet](https://github.com/open-mmlab/mmdetection.git), we recommend you to install the latest `mmdet` with pip.
See [here](https://pypi.org/project/mmdet/) for different versions of `mmdet`.

```shell
pip install mmdet==2.11.0
```

Optionally you can choose to install `mmdet` following the official [installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md).


e. Clone the mmocr repository.

```shell
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
```

f. Install build requirements and then install MMOCR.

```shell
pip install -r requirements.txt
pip install -v -e . # or "python setup.py build_ext --inplace"
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Full Set-up Script

Here is the full script for setting up mmocr with conda.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch

# install the latest mmcv-full
pip install mmcv-full==1.3.4

# install mmdetection
pip install mmdet==2.11.0

# install mmocr
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr

pip install -r requirements.txt
pip install -v -e .  # or "python setup.py build_ext --inplace"
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmocr/blob/master/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.5, CUDA 10.1
docker build -t mmocr docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmocr/data mmocr
```

## Prepare Datasets

It is recommended to symlink the dataset root to `mmocr/data`. Please refer to [datasets.md](datasets.md) to prepare your datasets.
If your folder structure is different, you may need to change the corresponding paths in config files.

The `mmocr` folder is organized as follows:
```
├── configs/
├── demo/
├── docker/
├── docs/
├── LICENSE
├── mmocr/
├── README.md
├── requirements/
├── requirements.txt
├── resources/
├── setup.cfg
├── setup.py
├── tests/
├── tools/
```
