<a id="markdown-installation" name="installation"></a>
# Installation
<!-- TOC -->

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Installation Instructions](#step-by-step-installation-instructions)
  - [Full Set-up Script](#full-set-up-script)
  - [Another option: Docker Image](#another-option-docker-image)
  - [Prepare Datasets](#prepare-datasets)

<!-- /TOC -->
<a id="markdown-prerequisites" name="prerequisites"></a>
## Prerequisites

- Linux (Windows is not officially supported)
- Python 3.7
- PyTorch 1.5
- torchvision 0.6.0
- CUDA 10.1
- NCCL 2
- GCC 5.4.0 or higher
- [mmcv](https://github.com/open-mmlab/mmcv) 1.2.6

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04
- CUDA: 10.1
- GCC(G++): 5.4.0
- mmcv 1.2.6
- PyTorch 1.5
- torchvision 0.6.0

MMOCR depends on Pytorch and mmdetection v2.9.0.

<a id="markdown-step-by-step-installation-instructions" name="step-by-step-installation-instructions"></a>
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

`E.g. 1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

```python
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

```python
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt package,
you can use more CUDA versions such as 9.0.

c. Create a folder called `code` and clone the mmcv repository into it.

```shell
mkdir code
cd code
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout -b v1.2.6 v1.2.6
pip install -r requirements.txt
MMCV_WITH_OPS=1 pip install -v -e .
```

d. Clone the mmdetection repository into it. The mmdetection repo is separate from the mmcv repo in `code`.

```shell
cd ..
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout -b v2.9.0 v2.9.0
pip install -r requirements.txt
pip install -v -e .
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Note that we have tested mmdetection v2.9.0 only. Other versions might be incompatible.

e. Clone the mmocr repository into it. The mmdetection repo is separate from the mmcv and mmdetection repo in `code`.

```shell
cd ..
git clone git@gitlab.sz.sensetime.com:kuangzhh/mmocr.git
cd mmocr
```

f. Install build requirements and then install MMOCR.

```shell
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py build_ext --inplace"
export PYTHONPATH=$(pwd):$PYTHONPATH
```

<a id="markdown-full-set-up-script" name="full-set-up-script"></a>
## Full Set-up Script

Here is the full script for setting up mmocr with conda.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch

# install mmcv
mkdir code
cd code
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv # code/mmcv
git checkout -b v1.2.6 v1.2.6
pip install -r requirements.txt
MMCV_WITH_OPS=1 pip install -v -e .

# install mmdetection
cd .. # exit to code
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection # code/mmdetection
git checkout -b v2.9.0 v2.9.0
pip install -r requirements.txt
pip install -v -e .
export PYTHONPATH=$(pwd):$PYTHONPATH

# install mmocr
cd ..
git clone git@gitlab.sz.sensetime.com:kuangzhh/mmocr.git
cd mmocr # code/mmocr

pip install -r requirements.txt
pip install -v -e .  # or "python setup.py build_ext --inplace"
export PYTHONPATH=$(pwd):$PYTHONPATH
```

<a id="markdown-another-option-docker-image" name="another-option-docker-image"></a>
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

<a id="markdown-prepare-datasets" name="prepare-datasets"></a>
## Prepare Datasets

It is recommended to symlink the dataset root to `mmocr/data`. Please refer to [datasets.md](datasets.md) to prepare your datasets.
If your folder structure is different, you may need to change the corresponding paths in config files.

The `mmocr` folder is organized as follows:
```
mmocr
.
├── configs
│   ├── _base_
│   ├── kie
│   ├── textdet
│   └── textrecog
├── demo
│   ├── demo_text_det.jpg
│   ├── demo_text_recog.jpg
│   ├── image_demo.py
│   └── webcam_demo.py
├── docs
│   ├── api.rst
│   ├── changelog.md
│   ├── code_of_conduct.md
│   ├── conf.py
│   ├── contributing.md
│   ├── datasets.md
│   ├── getting_started.md
│   ├── index.rst
│   ├── install.md
│   ├── make.bat
│   ├── Makefile
│   ├── merge_docs.sh
│   ├── requirements.txt
│   ├── res
│   ├── stats.py
│   └── technical_details.md
├── LICENSE
├── mmocr
│   ├── apis
│   ├── core
│   ├── datasets
│   ├── __init__.py
│   ├── models
│   ├── utils
│   └── version.py
├── README.md
├── requirements
│   ├── build.txt
│   ├── docs.txt
│   ├── optional.txt
│   ├── readthedocs.txt
│   ├── runtime.txt
│   └── tests.txt
├── requirements.txt
├── resources
│   ├── illustration.jpg
│   └── mmocr-logo.png
├── setup.cfg
├── setup.py
├── tests
│   ├── data
│   ├── test_dataset
│   ├── test_metrics
│   ├── test_models
│   ├── test_tools
│   └── test_utils
├── tmp.txt
└── tools
    ├── data
    ├── dist_test.sh
    ├── dist_train.sh
    ├── ocr_test_imgs.py
    ├── ocr_test_imgs.sh
    ├── publish_model.py
    ├── slurm_test.sh
    ├── slurm_train.sh
    ├── test_imgs.py
    ├── test_imgs.sh
    ├── test.py
    └── train.py
```

The icdar2017 official annotations can be converted into the coco format that mmocr supports using `code/mmocr/tools/data_converter/icdar_converter.py`.
