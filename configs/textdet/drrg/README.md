# DRRG

> [Deep relational reasoning graph network for arbitrary shape text detection](https://arxiv.org/abs/2003.07493)

<!-- [ALGORITHM] -->

## Abstract

Arbitrary shape text detection is a challenging task due to the high variety and complexity of scenes texts. In this paper, we propose a novel unified relational reasoning graph network for arbitrary shape text detection. In our method, an innovative local graph bridges a text proposal model via Convolutional Neural Network (CNN) and a deep relational reasoning network via Graph Convolutional Network (GCN), making our network end-to-end trainable. To be concrete, every text instance will be divided into a series of small rectangular components, and the geometry attributes (e.g., height, width, and orientation) of the small components will be estimated by our text proposal model. Given the geometry attributes, the local graph construction model can roughly establish linkages between different text components. For further reasoning and deducing the likelihood of linkages between the component and its neighbors, we adopt a graph-based network to perform deep relational reasoning on local graphs. Experiments on public available datasets demonstrate the state-of-the-art performance of our method.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142791777-f282300a-fb83-4b5a-a7d4-29f308949f11.png"/>
</div>

## Results and models

### CTW1500

|                       Method                       |       BackBone       | Pretrained Model | Training set  |   Test set   | #epochs | Test size | Precision | Recall | Hmean  |                       Download                       |
| :------------------------------------------------: | :------------------: | :--------------: | :-----------: | :----------: | :-----: | :-------: | :-------: | :----: | :----: | :--------------------------------------------------: |
| [DRRG](/configs/textdet/drrg/drrg_resnet50_fpn-unet_1200e_ctw1500.py) |       ResNet50       |        -         | CTW1500 Train | CTW1500 Test |  1200   |    640    |  0.8775   | 0.8179 | 0.8467 | [model](https://download.openmmlab.com/mmocr/textdet/drrg/drrg_resnet50_fpn-unet_1200e_ctw1500/drrg_resnet50_fpn-unet_1200e_ctw1500_20220827_105233-d5c702dd.pth) \\ [log](https://download.openmmlab.com/mmocr/textdet/drrg/drrg_resnet50_fpn-unet_1200e_ctw1500/20220827_105233.log) |
| [DRRG_oclip-r50](/configs/textdet/drrg/drrg_oclip-resnet50_fpn-unet_1200e_ctw1500.py) | [oCLIP-ResNet50](<>) |        -         | CTW1500 Train | CTW1500 Test |  1200   |           |           |        |        |               [model](<>) \\ [log](<>)               |

## Citation

```bibtex
@article{zhang2020drrg,
  title={Deep relational reasoning graph network for arbitrary shape text detection},
  author={Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Liu, Chang and Yang, Chun and Wang, Hongfa and Yin, Xu-Cheng},
  booktitle={CVPR},
  pages={9699-9708},
  year={2020}
}
```
