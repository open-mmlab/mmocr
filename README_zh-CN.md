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

最新的月度版本 v0.6.0 在 2022.05.05 发布。

1. 支持了两个新模型：[MASTER](https://arxiv.org/abs/1910.02562) 和 [DBNet++](https://arxiv.org/abs/2202.10304)
2. 新支持了三个数据集：LSVT, RCTW 和 HierText
3. 支持从 LMDB 数据集读取数据 [查看文档](https://mmocr.readthedocs.io/en/latest/tools.html#convert-text-recognition-dataset-to-lmdb-format)
4. 提供了 Labelme 数据格式转换脚本，可使 MMOCR 支持由 Labelme 标注的数据

阅读[更新日志](https://mmocr.readthedocs.io/en/latest/changelog.html)以获取更多信息。

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

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)，或通过添加微信“Open小喵Lab”加入官方交流微信群。

<div align="center">
<img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/zhihu_qrcode.jpg" height="400" />  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/qq_group_qrcode.jpg" height="400" />  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/wechat_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
