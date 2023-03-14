# ABCNet v2: Adaptive Bezier-Curve Network for Real-time End-to-end Text Spotting

<div>
<a href="https://arxiv.org/abs/2105.03620">[arXiv paper]</a>
<a href="https://ieeexplore.ieee.org/document/9525302">[TPAMI paper]</a>
</div>

## Description

This is an implementation of [ABCNetV2](https://github.com/aim-uofa/AdelaiDet) based on [MMOCR](https://github.com/open-mmlab/mmocr/tree/dev-1.x), [MMCV](https://github.com/open-mmlab/mmcv), and [MMEngine](https://github.com/open-mmlab/mmengine).

**ABCNetV2**  contributions are four-fold: 1) For the first time, we adaptively fit arbitrarily-shaped text by a parameterized Bezier curve, which, compared with segmentation-based methods, can not only provide structured output but also controllable representation. 2) We design a novel BezierAlign layer for extracting accurate convolution features of a text instance of arbitrary shapes, significantly improving the precision of recognition over previous methods. 3) Different from previous methods, which often suffer from complex post-processing and sensitive hyper-parameters, our ABCNet v2 maintains a simple pipeline with the only post-processing non-maximum suppression (NMS). 4) As the performance of text recognition closely depends on feature alignment, ABCNet v2 further adopts a simple yet effective coordinate convolution to encode the position of the convolutional filters, which leads to a considerable improvement with negligible computation overhead. Comprehensive experiments conducted on various bilingual (English and Chinese) benchmark datasets demonstrate that ABCNet v2 can achieve state-of-the-art performance while maintaining very high efficiency.

<center>
<img src="https://user-images.githubusercontent.com/24622904/213096846-2557e0ac-ca18-4c4f-88c1-569107f48f9b.png">
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

if the data is not in `ABCNet/`, you can link the data into `ABCNet/`:

```shell
# Linux
ln -s ${DataPath} $PYTHONPATH
# Windows PowerShell
New-Item -ItemType SymbolicLink -Path $env:PYTHONPATH -Name data  -Target ${DataPath}
```

### Testing commands

In the current directory, run the following command to test the model:

```bash
mim test mmocr config/abcnet_v2/abcnet-v2_resnet50_bifpn_500e_icdar2015.py --work-dir work_dirs/ --checkpoint ${CHECKPOINT_PATH}
```

## Results

Here we provide the baseline version of ABCNet with ResNet50 backbone.

To find more variants, please visit the [official model zoo](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BAText/README.md).

|         Name          | Pretrained Model | E2E-None-Hmean | det-Hmean |                                                                   Download                                                                   |
| :-------------------: | :--------------: | :------------: | :-------: | :------------------------------------------------------------------------------------------------------------------------------------------: |
| v2-icdar2015-finetune |    SynthText     |     0.6628     |  0.8886   | [model](https://download.openmmlab.com/mmocr/textspotting/abcnet-v2/abcnet-v2_resnet50_bifpn/abcnet-v2_resnet50_bifpn_500e_icdar2015-5e4cc7ed.pth) |

## Citation

If you find ABCNetV2 useful in your research or applications, please cite ABCNetV2 with the following BibTeX entry.

```BibTeX
@ARTICLE{9525302,
  author={Liu, Yuliang and Shen, Chunhua and Jin, Lianwen and He, Tong and Chen, Peng and Liu, Chongyu and Chen, Hao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={ABCNet v2: Adaptive Bezier-Curve Network for Real-time End-to-end Text Spotting},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3107437}}
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

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

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
