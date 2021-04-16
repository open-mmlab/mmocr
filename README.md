<div align="center">
  <img src="resources/mmocr-logo.png" width="500px"/>
</div>

## Introduction

English | [简体中文](README_zh-CN.md)

[![build](https://github.com/open-mmlab/mmocr/workflows/build/badge.svg)](https://github.com/open-mmlab/mmocr/actions)
[![docs](https://readthedocs.org/projects/mmocr/badge/?version=latest)](https://mmocr.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmocr/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmocr)
[![license](https://img.shields.io/github/license/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/mmocr.svg)](https://pypi.org/project/mmocr/)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/issues)

MMOCR is an open-source toolbox based on PyTorch and mmdetection for text detection, text recognition, and the corresponding downstream tasks including key information extraction. It is part of the [OpenMMLab](https://openmmlab.com/) project.

The main branch works with **PyTorch 1.5+**.

Documentation: https://mmocr.readthedocs.io/en/latest/.

<div align="left">
  <img src="resources/illustration.jpg"/>
</div>

### Major Features

- **Comprehensive Pipeline**

   The toolbox supports not only text detection and text recognition, but also their downstream tasks such as key information extraction.

- **Multiple Models**

  The toolbox supports a wide variety of state-of-the-art models for text detection, text recognition and key information extraction.

- **Modular Design**

  The modular design of MMOCR enables users to define their own optimizers, data preprocessors, and model components such as backbones, necks and heads as well as losses. Please refer to [getting_started.md](docs/getting_started.md) for how to construct a customized model.

- **Numerous Utilities**

  The toolbox provides a comprehensive set of utilities which can help users assess the performance of models. It includes visualizers which allow visualization of images, ground truths as well as predicted bounding boxes, and a validation tool for evaluating checkpoints during training.  It also includes data converters to demonstrate how to convert your own data to the annotation files which the toolbox supports.

## [Model Zoo](https://mmocr.readthedocs.io/en/latest/modelzoo.html)

Supported algorithms:

<details open>
<summary>(click to collapse)</summary>

- [x] [DBNet](configs/textdet/dbnet/README.md) (AAAI'2020)
- [x] [Mask R-CNN](configs/textdet/maskrcnn/README.md) (ICCV'2017)
- [x] [PANet](configs/textdet/panet/README.md) (ICCV'2019)
- [x] [PSENet](configs/textdet/psenet/README.md) (CVPR'2019)
- [x] [TextSnake](configs/textdet/textsnake/README.md) (ECCV'2018)
- [x] [CRNN](configs/textrecog/crnn/crnn_academic_dataset.py) (TPAMI'2016)
- [x] [NRTR](configs/textrecog/nrtr/README.md) (ICDAR'2019)
- [x] [RobustScanner](configs/textrecog/robust_scanner/README.md) (ECCV'2020)
- [x] [SAR](configs/textrecog/sar/README.md) (AAAI'2019)
- [x] [SegOCR](configs/bottom_up/higherhrnet/README.md) (Manuscript'2021)
- [x] [SDMG-R](configs/kie/sdmgr/README.md) (ArXiv'2021)

</details>

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmocr2021,
    title={MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding},
    author={MMOCR Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmocr}},
    year={2021}
}
```

## Changelog

v0.1.0 was released on 07/04/2021.

## Benchmark and Model Zoo

Please refer to [modelzoo.md](https://mmocr.readthedocs.io/en/latest/index.html) for more details.

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMOCR.

## Contributing

We appreciate all contributions to improve MMOCR. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guidelines.

## Acknowledgement

MMOCR is an open-source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We hope the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new OCR methods.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab's pose estimation toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image editing toolbox and benchmark.
