# ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network

<div>
<a href="https://arxiv.org/abs/2002.10200">[arXiv paper]</a>
<a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_ABCNet_Real-Time_Scene_Text_Spotting_With_Adaptive_Bezier-Curve_Network_CVPR_2020_paper.pdf">[CVPR paper]</a>
</div>

## Description

This is an implementation of [ABCNet](https://github.com/aim-uofa/AdelaiDet) based on [MMOCR](https://github.com/open-mmlab/mmocr/tree/dev-1.x), [MMCV](https://github.com/open-mmlab/mmcv), and [MMEngine](https://github.com/open-mmlab/mmengine).

**ABCNet** is a conceptually novel, efficient, and fully convolutional framework for text spotting, which address the problem by proposing the Adaptive Bezier-Curve Network (ABCNet). Our contributions are three-fold: 1) For the first time, we adaptively fit arbitrarily-shaped text by a parameterized Bezier curve. 2) We design a novel BezierAlign layer for extracting accurate convolution features of a text instance with arbitrary shapes, significantly improving the precision compared with previous methods. 3) Compared with standard bounding box detection, our Bezier curve detection introduces negligible computation overhead, resulting in superiority of our method in both efficiency and accuracy. Experiments on arbitrarily-shaped benchmark datasets, namely Total-Text and CTW1500, demonstrate that ABCNet achieves state-of-the-art accuracy, meanwhile significantly improving the speed. In particular, on Total-Text, our realtime version is over 10 times faster than recent state-of-the-art methods with a competitive recognition accuracy.

<center>
<img src="https://user-images.githubusercontent.com/24622904/205641295-d57c225f-4b2e-4954-b604-ba8c8afc23cb.png">
</center>

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Prerequisites

- Python 3.7
- PyTorch 1.6 or higher
- [MIM](https://github.com/open-mmlab/mim)
- [MMOCR](https://github.com/open-mmlab/mmocr)

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `ABCNet/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
# Linux
export PYTHONPATH=`pwd`:$PYTHONPATH
# Windows PowerShell
$env:PYTHONPATH=Get-Location
```

As of now, `BezierAlign` is not yet supported by MMCV, and we will use third-party MMCV with the implementation of `BezierAlign`. You will need to install it from the source code as follows:

```bash
git clone -b lkk/bezier_align https://github.com/Harold-lkk/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 MAX_JOBS=8 python setup.py develop
```

### Training commands

In the current directory, run the following command to train the model:

```bash
mim train mmocr config/abcnet/abcnet_resnet50_fpn.py --work-dir work_dirs/
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```bash
mim train mmocr config/abcnet/abcnet_resnet50_fpn.py --work-dir work_dirs/ --launcher pytorch --gpus 8
```

### Testing commands

In the current directory, run the following command to test the model:

```bash
mim test mmocr config/abcnet/abcnet_resnet50_fpn.py ${CHECKPOINT_PATH}
```

## Results

Here we provide the baseline version of ABCNet with ResNet50 backbone.

To find more variants, please visit the [official model zoo](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BAText/README.md).

|         Name          |                                  Pretrained Model                                  | E2E-None-Hmean | det-Hmean |                                  Download                                  |
| :-------------------: | :--------------------------------------------------------------------------------: | :------------: | :-------: | :------------------------------------------------------------------------: |
| v1-icdar2015-finetune | [SynthText](https://download.openmmlab.com/mmocr/textdet/abcnet/abcnet_resnet50_fpn_500e_icdar2015/abcnet_resnet50_fpn_pretrain-d060636c.pth) |     0.6127     |  0.8753   | [model](https://download.openmmlab.com/mmocr/textdet/abcnet/abcnet_resnet50_fpn_500e_icdar2015/abcnet_resnet50_fpn_500e_icdar2015-326ac6f4.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/abcnet/abcnet_resnet50_fpn_500e_icdar2015/20221210_170401.log) |

## Citation

If you find ABCNet useful in your research or applications, please cite ABCNet with the following BibTeX entry.

```BibTeX
@inproceedings{liu2020abcnet,
  title     =  {{ABCNet}: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network},
  author    =  {Liu, Yuliang and Chen, Hao and Shen, Chunhua and He, Tong and Jin, Lianwen and Wang, Liangwei},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.

OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.

Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.

A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmocr.registry.MODELS` and configurable via a config file. -->

  - [x] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [x] A full README

    <!-- As this template does. -->

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmocr/blob/76637a290507f151215d299707c57cea5120976e/mmocr/utils/polygon_utils.py#L80-L96) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmocr/blob/76637a290507f151215d299707c57cea5120976e/tests/test_utils/test_polygon_utils.py#L97-L106) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](/configs/textdet/dbnet/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
