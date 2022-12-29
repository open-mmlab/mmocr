# DBNetpp

> [Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/abs/2202.10304)

<!-- [ALGORITHM] -->

## Abstract

Recently, segmentation-based scene text detection methods have drawn extensive attention in the scene text detection field, because of their superiority in detecting the text instances of arbitrary shapes and extreme aspect ratios, profiting from the pixel-level descriptions. However, the vast majority of the existing segmentation-based approaches are limited to their complex post-processing algorithms and the scale robustness of their segmentation models, where the post-processing algorithms are not only isolated to the model optimization but also time-consuming and the scale robustness is usually strengthened by fusing multi-scale feature maps directly. In this paper, we propose a Differentiable Binarization (DB) module that integrates the binarization process, one of the most important steps in the post-processing procedure, into a segmentation network. Optimized along with the proposed DB module, the segmentation network can produce more accurate results, which enhances the accuracy of text detection with a simple pipeline. Furthermore, an efficient Adaptive Scale Fusion (ASF) module is proposed to improve the scale robustness by fusing features of different scales adaptively. By incorporating the proposed DB and ASF with the segmentation network, our proposed scene text detector consistently achieves state-of-the-art results, in terms of both detection accuracy and speed, on five standard benchmarks.

<div align=center>
<img src="https://user-images.githubusercontent.com/45810070/166850828-f1e48c25-4a0f-429d-ae54-6997ed25c062.png"/>
</div>

## Results and models

### ICDAR2015

|             Method             |             BackBone             |             Pretrained Model             |  Training set   |    Test set    | #epochs | Test size | Precision | Recall | Hmean  |             Download             |
| :----------------------------: | :------------------------------: | :--------------------------------------: | :-------------: | :------------: | :-----: | :-------: | :-------: | :----: | :----: | :------------------------------: |
| [DBNetpp_r50](/configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015.py) |             ResNet50             |                    -                     | ICDAR2015 Train | ICDAR2015 Test |  1200   |   1024    |  0.9079   | 0.8209 | 0.8622 | [model](https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015/dbnetpp_resnet50_fpnc_1200e_icdar2015_20221025_185550-013730aa.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015/20221025_185550.log) |
| [DBNetpp_r50dcn](/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py) |             ResNet50             | [Synthtext](/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_100k_synthtext.py) ([model](https://download.openmmlab.com/mmocr/textdet/dbnetpp/tmp_1.0_pretrain/dbnetpp_r50dcnv2_fpnc_100k_iter_synthtext-20220502-352fec8a.pth)) | ICDAR2015 Train | ICDAR2015 Test |  1200   |   1024    |  0.9116   | 0.8291 | 0.8684 | [model](https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015_20220829_230108-f289bd20.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015/20220829_230108.log) |
| [DBNetpp_r50-oclip](/configs/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py) | [ResNet50-oCLIP](https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth) |                    -                     | ICDAR2015 Train | ICDAR2015 Test |  1200   |   1024    |  0.9174   | 0.8609 | 0.8882 | [model](https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/20221101_124139.log) |

## Citation

```bibtex
@article{liao2022real,
    title={Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion},
    author={Liao, Minghui and Zou, Zhisheng and Wan, Zhaoyi and Yao, Cong and Bai, Xiang},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2022},
    publisher={IEEE}
}
```
