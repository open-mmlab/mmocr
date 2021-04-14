<div align="center">
  <img src="resources/mmocr-logo.png" width="500px"/>
</div>

## 简介

[English](/README.md) | 简体中文

[![build](https://github.com/open-mmlab/mmocr/workflows/build/badge.svg)](https://github.com/open-mmlab/mmocr/actions)
[![docs](https://readthedocs.org/projects/mmocr/badge/?version=latest)](https://mmocr.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmocr/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmocr)
[![license](https://img.shields.io/github/license/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/mmocr.svg)](https://pypi.org/project/mmocr/)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/issues)

MMOCR是基于PyTorch和mmdetection的开源工具箱，用于文本检测，文本识别以及相应的下游任务，包括关键信息提取。 它是OpenMMLab项目的一部分。

主分支目前支持 **PyTorch 1.5 以上**的版本。

文档：https://mmocr.readthedocs.io/en/latest/。

<div align="left">
  <img src="resources/illustration.jpg"/>
</div>

### 主要特性

-**综合管道**

   该工具箱不仅支持文本检测和文本识别，还支持其下游任务，例如关键信息提取。

-**多种模型**

  该工具箱支持用于文本检测，文本识别和关键信息提取的各种最新模型。

-**模块化设计**

  MMOCR的模块化设计使用户可以定义自己的优化器，数据预处理器和模型组件，例如主干模块，颈部模块和头部模块以及损失函数。有关如何构建自定义模型的信息，请参考[快速入门](docs/getting_started.md)。

-**众多实用工具**

  该工具箱提供了一套全面的实用程序，可以帮助用户评估模型的性能。它包括可对图像，基准真相以及预测的边界框进行可视化的可视化工具，以及用于在训练过程中评估检查点的验证工具。它还包括数据转换器，以演示如何将用户自建的数据转换为工具箱支持的注释文件。
## [模型库](https://mmocr.readthedocs.io/en/latest/modelzoo.html)

支持的算法：

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

## 开源许可证

该项目采用 [Apache 2.0 license](LICENSE) 开源许可证。 

## 引用

如果您发现此项目对您的研究有用，请考虑引用：

```bibtex
@misc{mmocr2021,
    title={MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding},
    author={MMOCR Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmocr}},
    year={2021}
}
```

## 更新日志

最新的月度版本 v0.1.0 在 2021.04.07 发布。

## 基准测试和模型库

测试结果和模型可以在[模型库](https://mmocr.readthedocs.io/en/latest/index.html)中找到。

## 安装

请参考[安装文档](docs/install.md)进行安装。

## 快速入门

请参考[快速入门](docs/getting_started.md)文档学习 MMOCR 的基本使用。 

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMOCR 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

MMOCR 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望此工具箱和基准测试可以通过提供灵活的工具箱来重新实现现有方法并开发用户自建的新OCR方法，从而为不断发展的研究社区服务。

## OpenMMLab 的其他项目


- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱与测试基准
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用3D目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包.
