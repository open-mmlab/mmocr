<div align="center">
  <img src="resources/mmocr-logo.png" width="500px"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![build](https://github.com/open-mmlab/mmocr/workflows/build/badge.svg)](https://github.com/open-mmlab/mmocr/actions)
[![docs](https://readthedocs.org/projects/mmocr/badge/?version=latest)](https://mmocr.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmocr/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmocr)
[![license](https://img.shields.io/github/license/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/mmocr.svg)](https://pypi.org/project/mmocr/)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmocr.svg)](https://github.com/open-mmlab/mmocr/issues)
<a href="https://console.tiyaro.ai/explore?q=mmocr&pub=mmocr"> <img src="https://tiyaro-public-docs.s3.us-west-2.amazonaws.com/assets/try_on_tiyaro_badge.svg"></a>

[📘文档](https://mmocr.readthedocs.io/zh_CN/latest/) |
[🛠️安装](https://mmocr.readthedocs.io/zh_CN/latest/install.html) |
[👀模型库](https://mmocr.readthedocs.io/zh_CN/latest/modelzoo.html) |
[🆕更新日志](https://mmocr.readthedocs.io/zh_CN/latest/changelog.html) |
[🤔报告问题](https://github.com/open-mmlab/mmocr/issues/new/choose)

</div>

<div align="center">

[English](/README.md) | 简体中文

</div>

## 简介

MMOCR 是基于 PyTorch 和 mmdetection 的开源工具箱，专注于文本检测，文本识别以及相应的下游任务，如关键信息提取。 它是 OpenMMLab 项目的一部分。

主分支目前支持 **PyTorch 1.6 以上**的版本。

<div align="center">
  <img src="resources/illustration.jpg"/>
</div>

### 主要特性

-**全流程**

该工具箱不仅支持文本检测和文本识别，还支持其下游任务，例如关键信息提取。

-**多种模型**

该工具箱支持用于文本检测，文本识别和关键信息提取的各种最新模型。

-**模块化设计**

MMOCR 的模块化设计使用户可以定义自己的优化器，数据预处理器，模型组件如主干模块，颈部模块和头部模块，以及损失函数。有关如何构建自定义模型的信
息，请参考[快速入门](https://mmocr.readthedocs.io/zh_CN/latest/getting_started.html)。

-**众多实用工具**

该工具箱提供了一套全面的实用程序，可以帮助用户评估模型的性能。它包括可对图像，标注的真值以及预测结果进行可视化的可视化工具，以及用于在训练过程中评估模型的验证工具。它还包括数据转换器，演示了如何将用户自建的标注数据转换为 MMOCR 支持的标注文件。

## 最新进展

目前我们正同步维护稳定版 (0.6.3) 和预览版 (1.0.0) 的 MMOCR，但稳定版会在 2022 年末开始逐步停止维护。我们建议用户尽早升级至 [MMOCR 1.0](https://github.com/open-mmlab/mmocr/tree/1.x)，以享受到由新架构带来的更多新特性和更佳的性能表现。阅读我们的[维护计划](https://mmocr.readthedocs.io/zh_CN/dev-1.x/migration/overview.html)以了解更多信息。

### 💎 稳定版本

最新的月度版本 v0.6.3 在 2022.11.03 发布。

这个版本增强了推理脚本的稳定性，并修复了可能导致 TorchServe 运行错误的问题。

阅读[更新日志](https://mmocr.readthedocs.io/en/latest/changelog.html)以获取更多信息。

### 🌟 1.x 预览版本

全新的 **v1.0.0rc3** 版本已经在 2022.11.03 发布：

1. 我们发布了数个以 [oCLIP-ResNet](https://github.com/open-mmlab/mmocr/blob/1.x/configs/backbone/oclip/README.md) 为骨干网络的预训练模型。该骨干网络是一种以 [oCLIP](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880282.pdf) 技术训练的 ResNet 变体，可以显著提升检测模型的表现。

2. 准备数据集通常是一件很繁琐的事情，在 OCR 领域尤甚。我们推出了全新的 [Dataset Preparer](https://mmocr.readthedocs.io/en/dev-1.x/user_guides/data_prepare/dataset_preparer.html)，帮助大家脱离繁琐的手工作业，仅需一条命令即可自动准备好多个 OCR 常用数据集。同时，该组件也通过模块化的设计，极大地减少了未来支持新数据集的难度。

3. 架构升级：MMOCR 1.x 是基于 [MMEngine](https://github.com/open-mmlab/mmengine)，提供了一个通用的、强大的执行器，允许更灵活的定制，提供了统一的训练和测试入口。

4. 统一接口：MMOCR 1.x 统一了数据集、模型、评估和可视化的接口和内部逻辑。支持更强的扩展性。

5. 跨项目调用：受益于统一的设计，你可以使用其他OpenMMLab项目中实现的模型，如 MMDet。 我们提供了一个例子，说明如何通过 `MMDetWrapper` 使用 MMDetection 的 Mask R-CNN。查看我们的文档以了解更多细节。更多的包装器将在未来发布。

6. 更强的可视化：我们提供了一系列可视化工具， 用户现在可以更方便可视化数据。

7. 更多的文档和教程：我们增加了更多的教程，降低用户的学习门槛。详见[教程](https://mmocr.readthedocs.io/zh_CN/dev-1.x/)。

8. 一站式数据准备：准备数据集已经不再是难事。使用我们的 [Dataset Preparer](https://mmocr.readthedocs.io/zh_CN/dev-1.x/user_guides/data_prepare/dataset_preparer.html)，一行命令即可让多个数据集准备就绪。

可以在 [1.x 分支](https://github.com/open-mmlab/mmocr/tree/1.x) 获取更多新特性。欢迎试用并提出反馈。

## 安装

MMOCR 依赖 [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) 和 [MMDetection](https://github.com/open-mmlab/mmdetection)，以下是安装的简要步骤。
更详细的安装指南请参考 [安装文档](https://mmocr.readthedocs.io/zh_CN/latest/install.html)。

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
pip3 install -e .
```

## 快速入门

请参考[快速入门](https://mmocr.readthedocs.io/zh_CN/latest/getting_started.html)文档学习 MMOCR 的基本使用。

## [模型库](https://mmocr.readthedocs.io/en/latest/modelzoo.html)

支持的算法：

<details open>
<summary>文字检测</summary>

- [x] [DBNet](configs/textdet/dbnet/README.md) (AAAI'2020) / [DBNet++](configs/textdet/dbnetpp/README.md) (TPAMI'2022)
- [x] [Mask R-CNN](configs/textdet/maskrcnn/README.md) (ICCV'2017)
- [x] [PANet](configs/textdet/panet/README.md) (ICCV'2019)
- [x] [PSENet](configs/textdet/psenet/README.md) (CVPR'2019)
- [x] [TextSnake](configs/textdet/textsnake/README.md) (ECCV'2018)
- [x] [DRRG](configs/textdet/drrg/README.md) (CVPR'2020)
- [x] [FCENet](configs/textdet/fcenet/README.md) (CVPR'2021)

</details>

<details open>
<summary>文字识别</summary>

- [x] [ABINet](configs/textrecog/abinet/README.md) (CVPR'2021)
- [x] [CRNN](configs/textrecog/crnn/README.md) (TPAMI'2016)
- [x] [MASTER](configs/textrecog/master/README.md) (PR'2021)
- [x] [NRTR](configs/textrecog/nrtr/README.md) (ICDAR'2019)
- [x] [RobustScanner](configs/textrecog/robust_scanner/README.md) (ECCV'2020)
- [x] [SAR](configs/textrecog/sar/README.md) (AAAI'2019)
- [x] [SATRN](configs/textrecog/satrn/README.md) (CVPR'2020 Workshop on Text and Documents in the Deep Learning Era)
- [x] [SegOCR](configs/textrecog/seg/README.md) (Manuscript'2021)

</details>

<details open>
<summary>关键信息提取</summary>

- [x] [SDMG-R](configs/kie/sdmgr/README.md) (ArXiv'2021)

</details>

<details open>
<summary>命名实体识别</summary>

- [x] [Bert-Softmax](configs/ner/bert_softmax/README.md) (NAACL'2019)

</details>

请点击[模型库](https://mmocr.readthedocs.io/en/latest/modelzoo.html)查看更多关于上述算法的详细信息。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMOCR 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

MMOCR 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望此工具箱可以帮助大家来复现已有的方法和开发新的方法，从而为研究社区贡献力量。

## 引用

如果您发现此项目对您的研究有用，请考虑引用：

```bibtex
@article{mmocr2021,
    title={MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding},
    author={Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin, Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang, Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua},
    journal= {arXiv preprint arXiv:2108.06543},
    year={2021}
}
```

## 开源许可证

该项目采用 [Apache 2.0 license](LICENSE) 开源许可证。

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具箱
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱与测试基准
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://r.vansin.top/?r=join-qq)，或通过添加微信“Open小喵Lab”加入官方交流微信群。

<div align="center">
<img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/zhihu_qrcode.jpg" height="400" />  <img src="https://cdn.vansin.top/OpenMMLab/q3.png" height="400" />  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/wechat_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
