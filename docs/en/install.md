# Installation

## Prerequisites

- Linux (Windows is not officially supported)
- Python 3.7
- PyTorch 1.6 or higher
- torchvision 0.7.0
- CUDA 10.1
- NCCL 2
- GCC 5.4.0 or higher
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) >= 1.3.8
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/#installation) >= 2.14.0

We have tested the following versions of OS and software:

- OS: Ubuntu 16.04
- CUDA: 10.1
- GCC(G++): 5.4.0
- MMCV 1.3.8
- MMDetection 2.14.0
- PyTorch 1.6.0
- torchvision 0.7.0

MMOCR depends on PyTorch and mmdetection.

## Step-by-Step Installation Instructions

a. Create a Conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

:::{note}
Make sure that your compilation CUDA version and runtime CUDA version matches.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).
:::


c. Install [mmcv](https://github.com/open-mmlab/mmcv), we recommend you to install the pre-build mmcv as below.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace ``{cu_version}`` and ``{torch_version}`` in the url with your desired one. For example, to install the latest ``mmcv-full`` with CUDA 11 and PyTorch 1.7.0, use the following command:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```
:::{note}
mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well.

    ```
    # We can ignore the micro version of PyTorch
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
    ```
:::
:::{note}
Note that mmocr 0.2.1 or later requires mmcv 1.3.8 or later.

If it compiles during installation, then please check that the CUDA version and PyTorch version **exactly** matches the version in the `mmcv-full` installation command. For example, PyTorch 1.7.0 and 1.7.1 are treated differently.

See official [installation guide](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.
:::

:::{warning}
You need to run `pip uninstall mmcv` first if you have `mmcv` installed. If `mmcv` and `mmcv-full` are both installed, there will be `ModuleNotFoundError`.
:::

d. Install [mmdet](https://github.com/open-mmlab/mmdetection), we recommend you to install the latest `mmdet` with pip.
See [here](https://pypi.org/project/mmdet/) for different versions of `mmdet`.

```shell
pip install mmdet
```

Optionally you can choose to install `mmdet` following the official [installation guide](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md).


e. Clone the MMOCR repository.

```shell
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
```

f. Install build requirements and then install MMOCR.

```shell
pip install -r requirements.txt
pip install -v -e . # or "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Full Set-up Script

Here is the full script for setting up MMOCR with Conda.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

# install the latest mmcv-full
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# install mmdetection
pip install mmdet

# install mmocr
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr

pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmocr/blob/master/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
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
