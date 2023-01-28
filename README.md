<div align="center">
  <img src="resources/mmocr-logo.png" width="500px"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![build](https://github.com/open-mmlab/mmocr/workflows/build/badge.svg)](https://github.com/open-mmlab/mmocr/actions)
[![docs](https://readthedocs.org/projects/mmocr/badge/?version=dev-1.x)](https://mmocr.readthedocs.io/en/dev-1.x/?badge=dev-1.x)
[![codecov](https://codecov.io/gh/open-mmlab/mmocr/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmocr)
[![license](https://img.shields.io/github/license/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/mmocr.svg)](https://pypi.org/project/mmocr/)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/issues)
<a href="https://console.tiyaro.ai/explore?q=mmocr&pub=mmocr"> <img src="https://tiyaro-public-docs.s3.us-west-2.amazonaws.com/assets/try_on_tiyaro_badge.svg"></a>

[📘Documentation](https://mmocr.readthedocs.io/en/dev-1.x/) |
[🛠️Installation](https://mmocr.readthedocs.io/en/dev-1.x/get_started/install.html) |
[👀Model Zoo](https://mmocr.readthedocs.io/en/dev-1.x/modelzoo.html) |
[🆕Update News](https://mmocr.readthedocs.io/en/dev-1.x/notes/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmocr/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMOCR is an open-source toolbox based on PyTorch and mmdetection for text detection, text recognition, and the corresponding downstream tasks including key information extraction. It is part of the [OpenMMLab](https://openmmlab.com/) project.

The main branch works with **PyTorch 1.6+**.

<div align="center">
  <img src="https://user-images.githubusercontent.com/24622904/187838618-1fdc61c0-2d46-49f9-8502-976ffdf01f28.png"/>
</div>

### Major Features

- **Comprehensive Pipeline**

  The toolbox supports not only text detection and text recognition, but also their downstream tasks such as key information extraction.

- **Multiple Models**

  The toolbox supports a wide variety of state-of-the-art models for text detection, text recognition and key information extraction.

- **Modular Design**

  The modular design of MMOCR enables users to define their own optimizers, data preprocessors, and model components such as backbones, necks and heads as well as losses. Please refer to [Overview](https://mmocr.readthedocs.io/en/dev-1.x/get_started/overview.html) for how to construct a customized model.

- **Numerous Utilities**

  The toolbox provides a comprehensive set of utilities which can help users assess the performance of models. It includes visualizers which allow visualization of images, ground truths as well as predicted bounding boxes, and a validation tool for evaluating checkpoints during training.  It also includes data converters to demonstrate how to convert your own data to the annotation files which the toolbox supports.

## Latest Updates

**The default branch has been switched to `1.x` from `main`, and we encourage
users to migrate to the latest version, though it comes with some cost. Please refer to [Migration Guide](https://mmocr.readthedocs.io/en/dev-1.x/migration/overview.html) for more
details.**

v1.0.0rc5 was released in 2023-01-06.

1. Two models, Aster and SVTR, are added to our model zoo. The full implementation of ABCNet is also available now.
2. Dataset Preparer supports 5 more datasets: CocoTextV2, FUNSD, TextOCR, NAF, SROIE.
3. We have 4 more text recognition transforms, and two more helper transforms.
4. The transform, `FixInvalidPolygon`, is getting smarter at dealing with invalid polygons, and now capable of handling more weird annotations. As a result, a complete training cycle on TotalText dataset can be performed bug-free. The weights of DBNet and FCENet pretrained on TotalText are also released.

Read [Changelog](https://mmocr.readthedocs.io/en/dev-1.x/notes/changelog.html) for more details!

## What's New in MMOCR 1.0

1. **New engines**. MMOCR 1.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a general and powerful runner that allows more flexible customizations and significantly simplifies the entrypoints of high-level interfaces.

2. **Unified interfaces**. As a part of the OpenMMLab 2.0 projects, MMOCR 1.x unifies and refactors the interfaces and internal logics of train, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logics to allow the emergence of multi-task/modality algorithms.

3. **Cross project calling**. Benefiting from the unified design, you can use the models implemented in other OpenMMLab projects, such as MMDet. We provide an example of how to use MMDetection's Mask R-CNN through `MMDetWrapper`. Check our documents for more details. More wrappers will be released in the future.

4. **Stronger visualization**. We provide a series of useful tools which are mostly based on brand-new visualizers. As a result, it is more convenient for the users to explore the models and datasets now.

5. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmocr.readthedocs.io/en/dev-1.x/).

6. **One-stop Dataset Preparaion**. Multiple datasets are instantly ready with only one line of command, via our [Dataset Preparer](https://mmocr.readthedocs.io/en/dev-1.x/user_guides/data_prepare/dataset_preparer.html).

7. **Embracing more `projects/`**: We now introduce `projects/` folder, where some experimental features, frameworks and models can be placed, only needed to satisfy the minimum requirement on the code quality. Everyone is welcome to post their implementation of any great ideas in this folder! Learn more from our [example project](https://github.com/open-mmlab/mmocr/blob/dev-1.x/projects/example_project/).

8. **More models**. MMOCR 1.0 supports more tasks and more state-of-the-art models!

## Installation

MMOCR depends on [PyTorch](https://pytorch.org/), [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmocr.readthedocs.io/en/dev-1.x/get_started/install.html) for more detailed instruction.

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
mim install 'mmdet>=3.0.0rc0'
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
git checkout 1.x
pip3 install -e .
```

## Get Started

Please see [Quick Run](https://mmocr.readthedocs.io/en/dev-1.x/get_started/quick_run.html) for the basic usage of MMOCR.

## [Model Zoo](https://mmocr.readthedocs.io/en/dev-1.x/modelzoo.html)

Supported algorithms:

<details open>
<summary>BackBone</summary>

- [x] [oCLIP](configs/backbone/oclip/README.md) (ECCV'2022)

</details>

<details open>
<summary>Text Detection</summary>

- [x] [DBNet](configs/textdet/dbnet/README.md) (AAAI'2020) / [DBNet++](configs/textdet/dbnetpp/README.md) (TPAMI'2022)
- [x] [Mask R-CNN](configs/textdet/maskrcnn/README.md) (ICCV'2017)
- [x] [PANet](configs/textdet/panet/README.md) (ICCV'2019)
- [x] [PSENet](configs/textdet/psenet/README.md) (CVPR'2019)
- [x] [TextSnake](configs/textdet/textsnake/README.md) (ECCV'2018)
- [x] [DRRG](configs/textdet/drrg/README.md) (CVPR'2020)
- [x] [FCENet](configs/textdet/fcenet/README.md) (CVPR'2021)

</details>

<details open>
<summary>Text Recognition</summary>

- [x] [ABINet](configs/textrecog/abinet/README.md) (CVPR'2021)
- [x] [ASTER](configs/textrecog/aster/README.md) (TPAMI'2018)
- [x] [CRNN](configs/textrecog/crnn/README.md) (TPAMI'2016)
- [x] [MASTER](configs/textrecog/master/README.md) (PR'2021)
- [x] [NRTR](configs/textrecog/nrtr/README.md) (ICDAR'2019)
- [x] [RobustScanner](configs/textrecog/robust_scanner/README.md) (ECCV'2020)
- [x] [SAR](configs/textrecog/sar/README.md) (AAAI'2019)
- [x] [SATRN](configs/textrecog/satrn/README.md) (CVPR'2020 Workshop on Text and Documents in the Deep Learning Era)
- [x] [SVTR](configs/textrecog/svtr/README.md) (IJCAI'2022)

</details>

<details open>
<summary>Key Information Extraction</summary>

- [x] [SDMG-R](configs/kie/sdmgr/README.md) (ArXiv'2021)

</details>

<details open>
<summary>Text Spotting</summary>

- [x] [ABCNet](projects/ABCNet/README.md) (CVPR'2020)

</details>

Please refer to [model_zoo](https://mmocr.readthedocs.io/en/dev-1.x/modelzoo.html) for more details.

## Contributing

We appreciate all contributions to improve MMOCR. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guidelines.

## Acknowledgement

MMOCR is an open-source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We hope the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new OCR methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@article{mmocr2021,
    title={MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding},
    author={Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin, Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang, Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua},
    journal= {arXiv preprint arXiv:2108.06543},
    year={2021}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.

## Welcome to the OpenMMLab community

Scan the QR code below to follow the OpenMMLab team's [**Zhihu Official Account**](https://www.zhihu.com/people/openmmlab) and join the OpenMMLab team's [**QQ Group**](https://jq.qq.com/?_wv=1027&k=aCvMxdr3), or join the official communication WeChat group by adding the WeChat, or join our [**Slack**](https://join.slack.com/t/mmocrworkspace/shared_invite/zt-1ifqhfla8-yKnLO_aKhVA2h71OrK8GZw)

<div align="center">
<img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/zhihu_qrcode.jpg" height="400" />  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/qq_group_qrcode.jpg" height="400" />  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/wechat_qrcode.jpg" height="400" />
</div>

We will provide you with the OpenMMLab community

- 📢 share the latest core technologies of AI frameworks
- 💻 Explaining PyTorch common module source Code
- 📰 News related to the release of OpenMMLab
- 🚀 Introduction of cutting-edge algorithms developed by OpenMMLab
  🏃 Get the more efficient answer and feedback
- 🔥 Provide a platform for communication with developers from all walks of life

The OpenMMLab community looks forward to your participation! 👬
