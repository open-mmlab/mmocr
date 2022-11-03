# PSENet

> [Shape robust text detection with progressive scale expansion network](https://arxiv.org/abs/1903.12473)

<!-- [ALGORITHM] -->

## Abstract

Scene text detection has witnessed rapid progress especially with the recent development of convolutional neural networks. However, there still exists two challenges which prevent the algorithm into industry applications. On the one hand, most of the state-of-art algorithms require quadrangle bounding box which is in-accurate to locate the texts with arbitrary shape. On the other hand, two text instances which are close to each other may lead to a false detection which covers both instances. Traditionally, the segmentation-based approach can relieve the first problem but usually fail to solve the second challenge. To address these two challenges, in this paper, we propose a novel Progressive Scale Expansion Network (PSENet), which can precisely detect text instances with arbitrary shapes. More specifically, PSENet generates the different scale of kernels for each text instance, and gradually expands the minimal scale kernel to the text instance with the complete shape. Due to the fact that there are large geometrical margins among the minimal scale kernels, our method is effective to split the close text instances, making it easier to use segmentation-based methods to detect arbitrary-shaped text instances. Extensive experiments on CTW1500, Total-Text, ICDAR 2015 and ICDAR 2017 MLT validate the effectiveness of PSENet. Notably, on CTW1500, a dataset full of long curve texts, PSENet achieves a F-measure of 74.3% at 27 FPS, and our best F-measure (82.2%) outperforms state-of-art algorithms by 6.6%. The code will be released in the future.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142795864-9b455b10-8a19-45bb-aeaf-4b733f341afc.png"/>
</div>

## Results and models

### CTW1500

|                       Method                       |       Backbone       | Pretrained Model | Training set  |   Test set   | #epochs | Test size | Precision | Recall | Hmean  |                       Download                       |
| :------------------------------------------------: | :------------------: | :--------------: | :-----------: | :----------: | :-----: | :-------: | :-------: | :----: | :----: | :--------------------------------------------------: |
| [PSENet](/configs/textdet/psenet/psenet_resnet50_fpnf_600e_ctw1500.py) |       ResNet50       |        -         | CTW1500 Train | CTW1500 Test |   600   |   1280    |  0.7705   | 0.7883 | 0.7793 | [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_resnet50_fpnf_600e_ctw1500/psenet_resnet50_fpnf_600e_ctw1500_20220825_221459-7f974ac8.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_resnet50_fpnf_600e_ctw1500/20220825_221459.log) |
| [PSENet_oclip-r50](/configs/textdet/psenet/psenet_oclip-resnet50_fpnf_600e_ctw1500.py) | [oCLIP-ResNet50](<>) |        -         | CTW1500 Train | CTW1500 Test |   600   |   1280    |           |        |        |               [model](<>) \| [log](<>)               |

### ICDAR2015

|                        Method                        |       Backbone       | Pretrained Model | Training set | Test set  | #epochs | Test size | Precision | Recall | Hmean  |                        Download                        |
| :--------------------------------------------------: | :------------------: | :--------------: | :----------: | :-------: | :-----: | :-------: | :-------: | :----: | :----: | :----------------------------------------------------: |
| [PSENet](/configs/textdet/psenet/psenet_resnet50_fpnf_600e_icdar2015.py) |       ResNet50       |        -         |  IC15 Train  | IC15 Test |   600   |   2240    |  0.8396   | 0.7636 | 0.7998 | [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_resnet50_fpnf_600e_icdar2015/psenet_resnet50_fpnf_600e_icdar2015_20220825_222709-b6741ec3.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_resnet50_fpnf_600e_icdar2015/20220825_222709.log) |
| [PSENet_oclip-r50](/configs/textdet/psenet/psenet_oclip-resnet50_fpnf_600e_icdar2015.py) | [oCLIP-ResNet50](<>) |        -         |  IC15 Train  | IC15 Test |   600   |   2240    |           |        |        |                [model](<>) \| [log](<>)                |

## Citation

```bibtex
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```
