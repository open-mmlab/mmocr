# Text Detection Models

## DBNetpp

[Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/abs/2202.10304)

<!-- [ALGORITHM] -->

### Abstract

Recently, segmentation-based scene text detection methods have drawn extensive attention in the scene text detection field, because of their superiority in detecting the text instances of arbitrary shapes and extreme aspect ratios, profiting from the pixel-level descriptions. However, the vast majority of the existing segmentation-based approaches are limited to their complex post-processing algorithms and the scale robustness of their segmentation models, where the post-processing algorithms are not only isolated to the model optimization but also time-consuming and the scale robustness is usually strengthened by fusing multi-scale feature maps directly. In this paper, we propose a Differentiable Binarization (DB) module that integrates the binarization process, one of the most important steps in the post-processing procedure, into a segmentation network. Optimized along with the proposed DB module, the segmentation network can produce more accurate results, which enhances the accuracy of text detection with a simple pipeline. Furthermore, an efficient Adaptive Scale Fusion (ASF) module is proposed to improve the scale robustness by fusing features of different scales adaptively. By incorporating the proposed DB and ASF with the segmentation network, our proposed scene text detector consistently achieves state-of-the-art results, in terms of both detection accuracy and speed, on five standard benchmarks.

<div align=center>
<img src="https://user-images.githubusercontent.com/45810070/166850828-f1e48c25-4a0f-429d-ae54-6997ed25c062.png"/>
</div>

### Results and models

#### ICDAR2015

|                  Method                  |                  Pretrained Model                   |  Training set   |    Test set    | ##epochs | Test size | Recall | Precision | Hmean |                  Download                   |
| :--------------------------------------: | :-------------------------------------------------: | :-------------: | :------------: | :------: | :-------: | :----: | :-------: | :---: | :-----------------------------------------: |
| [DBNetpp_r50dcn](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py) | [Synthtext](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_100k_synthtext.py) ([model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_100k_iter_synthtext-20220502-db297554.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_100k_iter_synthtext-20220502-db297554.log.json)) | ICDAR2015 Train | ICDAR2015 Test |   1200   |   1024    | 0.822  |   0.901   | 0.860 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.log.json) |

### Citation

```bibtex
@article{liao2022real,
    title={Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion},
    author={Liao, Minghui and Zou, Zhisheng and Wan, Zhaoyi and Yao, Cong and Bai, Xiang},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2022},
    publisher={IEEE}
}
```

## DBNet

[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

<!-- [ALGORITHM] -->

### Abstract

Recently, segmentation-based methods are quite popular in scene text detection, as the segmentation results can more accurately describe scene text of various shapes such as curve text. However, the post-processing of binarization is essential for segmentation-based detection, which converts probability maps produced by a segmentation method into bounding boxes/regions of text. In this paper, we propose a module named Differentiable Binarization (DB), which can perform the binarization process in a segmentation network. Optimized along with a DB module, a segmentation network can adaptively set the thresholds for binarization, which not only simplifies the post-processing but also enhances the performance of text detection. Based on a simple segmentation network, we validate the performance improvements of DB on five benchmark datasets, which consistently achieves state-of-the-art results, in terms of both detection accuracy and speed. In particular, with a light-weight backbone, the performance improvements by DB are significant so that we can look for an ideal tradeoff between detection accuracy and efficiency. Specifically, with a backbone of ResNet-18, our detector achieves an F-measure of 82.8, running at 62 FPS, on the MSRA-TD500 dataset.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142791306-0da6db2a-20a6-4a68-b228-64ff275f67b3.png"/>
</div>

### Results and models

#### ICDAR2015

|                  Method                  |                  Pretrained Model                   |  Training set   |    Test set    | ##epochs | Test size | Recall | Precision | Hmean |                  Download                   |
| :--------------------------------------: | :-------------------------------------------------: | :-------------: | :------------: | :------: | :-------: | :----: | :-------: | :---: | :-----------------------------------------: |
| [DBNet_r18](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py) |                      ImageNet                       | ICDAR2015 Train | ICDAR2015 Test |   1200   |    736    | 0.731  |   0.871   | 0.795 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.log.json) |
| [DBNet_r50dcn](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py) | [Synthtext](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth) | ICDAR2015 Train | ICDAR2015 Test |   1200   |   1024    | 0.814  |   0.868   | 0.840 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.log.json) |

### Citation

```bibtex
@article{Liao_Wan_Yao_Chen_Bai_2020,
    title={Real-Time Scene Text Detection with Differentiable Binarization},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
    year={2020},
    pages={11474-11481}}
```

## DRRG

[Deep relational reasoning graph network for arbitrary shape text detection](https://arxiv.org/abs/2003.07493)

<!-- [ALGORITHM] -->

### Abstract

Arbitrary shape text detection is a challenging task due to the high variety and complexity of scenes texts. In this paper, we propose a novel unified relational reasoning graph network for arbitrary shape text detection. In our method, an innovative local graph bridges a text proposal model via Convolutional Neural Network (CNN) and a deep relational reasoning network via Graph Convolutional Network (GCN), making our network end-to-end trainable. To be concrete, every text instance will be divided into a series of small rectangular components, and the geometry attributes (e.g., height, width, and orientation) of the small components will be estimated by our text proposal model. Given the geometry attributes, the local graph construction model can roughly establish linkages between different text components. For further reasoning and deducing the likelihood of linkages between the component and its neighbors, we adopt a graph-based network to perform deep relational reasoning on local graphs. Experiments on public available datasets demonstrate the state-of-the-art performance of our method.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142791777-f282300a-fb83-4b5a-a7d4-29f308949f11.png"/>
</div>

### Results and models

#### CTW1500

|                       Method                       | Pretrained Model | Training set  |   Test set   | ##epochs | Test size |    Recall     |   Precision   |     Hmean     |                       Download                        |
| :------------------------------------------------: | :--------------: | :-----------: | :----------: | :------: | :-------: | :-----------: | :-----------: | :-----------: | :---------------------------------------------------: |
| [DRRG](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/drrg/drrg_resnet50_fpn-unet_1200e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |   1200   |    640    | 0.822 (0.791) | 0.858 (0.862) | 0.840 (0.825) | [model](https://download.openmmlab.com/mmocr/textdet/drrg/drrg_r50_fpn_unet_1200e_ctw1500_20211022-fb30b001.pth) \\ [log](https://download.openmmlab.com/mmocr/textdet/drrg/20210511_234719.log) |

```{note}
We've upgraded our IoU backend from `Polygon3` to `shapely`. There are some performance differences for some models due to the backends' different logics to handle invalid polygons (more info [here](https://github.com/open-mmlab/mmocr/issues/465)). **New evaluation result is presented in brackets** and new logs will be uploaded soon.
```

### Citation

```bibtex
@article{zhang2020drrg,
  title={Deep relational reasoning graph network for arbitrary shape text detection},
  author={Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Liu, Chang and Yang, Chun and Wang, Hongfa and Yin, Xu-Cheng},
  booktitle={CVPR},
  pages={9699-9708},
  year={2020}
}
```

## FCENet

[Fourier Contour Embedding for Arbitrary-Shaped Text Detection](https://arxiv.org/abs/2104.10442)

<!-- [ALGORITHM] -->

### Abstract

One of the main challenges for arbitrary-shaped text detection is to design a good text instance representation that allows networks to learn diverse text geometry variances. Most of existing methods model text instances in image spatial domain via masks or contour point sequences in the Cartesian or the polar coordinate system. However, the mask representation might lead to expensive post-processing, while the point sequence one may have limited capability to model texts with highly-curved shapes. To tackle these problems, we model text instances in the Fourier domain and propose one novel Fourier Contour Embedding (FCE) method to represent arbitrary shaped text contours as compact signatures. We further construct FCENet with a backbone, feature pyramid networks (FPN) and a simple post-processing with the Inverse Fourier Transformation (IFT) and Non-Maximum Suppression (NMS). Different from previous methods, FCENet first predicts compact Fourier signatures of text instances, and then reconstructs text contours via IFT and NMS during test. Extensive experiments demonstrate that FCE is accurate and robust to fit contours of scene texts even with highly-curved shapes, and also validate the effectiveness and the good generalization of FCENet for arbitrary-shaped text detection. Furthermore, experimental results show that our FCENet is superior to the state-of-the-art (SOTA) methods on CTW1500 and Total-Text, especially on challenging highly-curved text subset.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142791859-1b0ebde4-b151-4c25-ba1b-f354bd8ddc8c.png"/>
</div>

### Results and models

#### CTW1500

|                       Method                       |     Backbone     | Pretrained Model | Training set  |   Test set   | ##epochs |  Test size  | Recall | Precision | Hmean  |                       Download                        |
| :------------------------------------------------: | :--------------: | :--------------: | :-----------: | :----------: | :------: | :---------: | :----: | :-------: | :----: | :---------------------------------------------------: |
| [FCENet](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/fcenet/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500.py) | ResNet50 + DCNv2 |     ImageNet     | CTW1500 Train | CTW1500 Test |   1500   | (736, 1080) | 0.8468 |  0.8532   | 0.8500 | [model](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500_20211022-e326d7ec.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/fcenet/20210511_181328.log.json) |

#### ICDAR2015

|                          Method                          | Backbone | Pretrained Model | Training set | Test set  | ##epochs |  Test size   | Recall | Precision | Hmean  |                          Download                          |
| :------------------------------------------------------: | :------: | :--------------: | :----------: | :-------: | :------: | :----------: | :----: | :-------: | :----: | :--------------------------------------------------------: |
| [FCENet](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/fcenet/fcenet_resnet50_fpn_1500e_icdar2015.py) | ResNet50 |     ImageNet     |  IC15 Train  | IC15 Test |   1500   | (2260, 2260) | 0.8243 |  0.8834   | 0.8528 | [model](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/fcenet/20210601_222655.log.json) |

### Citation

```bibtex
@InProceedings{zhu2021fourier,
      title={Fourier Contour Embedding for Arbitrary-Shaped Text Detection},
      author={Yiqin Zhu and Jianyong Chen and Lingyu Liang and Zhanghui Kuang and Lianwen Jin and Wayne Zhang},
      year={2021},
      booktitle = {CVPR}
      }
```

## Mask R-CNN

[Mask R-CNN](https://arxiv.org/abs/1703.06870)

<!-- [ALGORITHM] -->

### Abstract

We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142795605-dfdd5f69-e9cd-4b69-9c6b-6d8bded18e89.png"/>
</div>

### Results and models

#### CTW1500

|                           Method                            | Pretrained Model | Training set  |   Test set   | ##epochs | Test size | Recall | Precision | Hmean  |                            Download                            |
| :---------------------------------------------------------: | :--------------: | :-----------: | :----------: | :------: | :-------: | :----: | :-------: | :----: | :------------------------------------------------------------: |
| [MaskRCNN](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/maskrcnn/mask-rcnn_resnet50_fpn_160e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |   160    |   1600    | 0.7714 |  0.7272   | 0.7486 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.log.json) |

#### ICDAR2015

|                          Method                           | Pretrained Model |  Training set   |    Test set    | ##epochs | Test size | Recall | Precision | Hmean  |                           Download                           |
| :-------------------------------------------------------: | :--------------: | :-------------: | :------------: | :------: | :-------: | :----: | :-------: | :----: | :----------------------------------------------------------: |
| [MaskRCNN](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/maskrcnn/mask-rcnn_resnet50_fpn_160e_icdar2015.py) |     ImageNet     | ICDAR2015 Train | ICDAR2015 Test |   160    |   1920    | 0.8045 |  0.8530   | 0.8280 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.log.json) |

#### ICDAR2017

|                           Method                           | Pretrained Model |  Training set   |   Test set    | ##epochs | Test size | Recall | Precision | Hmean |                           Download                            |
| :--------------------------------------------------------: | :--------------: | :-------------: | :-----------: | :------: | :-------: | :----: | :-------: | :---: | :-----------------------------------------------------------: |
| [MaskRCNN](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/maskrcnn/mask-rcnn_resnet50_fpn_160e_icdar2017.py) |     ImageNet     | ICDAR2017 Train | ICDAR2017 Val |   160    |   1600    | 0.754  |   0.827   | 0.789 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.log.json) |

```{note}
We tuned parameters with the techniques in [Pyramid Mask Text Detector](https://arxiv.org/abs/1903.11800)
```

### Citation

```bibtex
@INPROCEEDINGS{8237584,
  author={K. {He} and G. {Gkioxari} and P. {Dollár} and R. {Girshick}},
  booktitle={2017 IEEE International Conference on Computer Vision (ICCV)},
  title={Mask R-CNN},
  year={2017},
  pages={2980-2988},
  doi={10.1109/ICCV.2017.322}}
```

## PANet

[Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network](https://arxiv.org/abs/1908.05900)

<!-- [ALGORITHM] -->

### Abstract

Scene text detection, an important step of scene text reading systems, has witnessed rapid development with convolutional neural networks. Nonetheless, two main challenges still exist and hamper its deployment to real-world applications. The first problem is the trade-off between speed and accuracy. The second one is to model the arbitrary-shaped text instance. Recently, some methods have been proposed to tackle arbitrary-shaped text detection, but they rarely take the speed of the entire pipeline into consideration, which may fall short in practical this http URL this paper, we propose an efficient and accurate arbitrary-shaped text detector, termed Pixel Aggregation Network (PAN), which is equipped with a low computational-cost segmentation head and a learnable post-processing. More specifically, the segmentation head is made up of Feature Pyramid Enhancement Module (FPEM) and Feature Fusion Module (FFM). FPEM is a cascadable U-shaped module, which can introduce multi-level information to guide the better segmentation. FFM can gather the features given by the FPEMs of different depths into a final feature for segmentation. The learnable post-processing is implemented by Pixel Aggregation (PA), which can precisely aggregate text pixels by predicted similarity vectors. Experiments on several standard benchmarks validate the superiority of the proposed PAN. It is worth noting that our method can achieve a competitive F-measure of 79.9% at 84.2 FPS on CTW1500.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142795741-0e1ea962-1596-47c2-8671-27bbe87d0df8.png"/>
</div>

### Results and models

#### CTW1500

|                       Method                       | Pretrained Model | Training set  |   Test set   | ##epochs | Test size |    Recall     |   Precision   |     Hmean     |                       Download                        |
| :------------------------------------------------: | :--------------: | :-----------: | :----------: | :------: | :-------: | :-----------: | :-----------: | :-----------: | :---------------------------------------------------: |
| [PANet](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/panet/panet_resnet18_fpem-ffm_600e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |   600    |    640    | 0.776 (0.717) | 0.838 (0.835) | 0.806 (0.801) | [model](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.log.json) |

#### ICDAR2015

|                      Method                       | Pretrained Model |  Training set   |    Test set    | ##epochs | Test size |    Recall    |  Precision   |     Hmean     |                       Download                       |
| :-----------------------------------------------: | :--------------: | :-------------: | :------------: | :------: | :-------: | :----------: | :----------: | :-----------: | :--------------------------------------------------: |
| [PANet](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/panet/panet_resnet18_fpem-ffm_600e_icdar2015.py) |     ImageNet     | ICDAR2015 Train | ICDAR2015 Test |   600    |    736    | 0.734 (0.74) | 0.856 (0.86) | 0.791 (0.795) | [model](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.log.json) |

```{note}
We've upgraded our IoU backend from `Polygon3` to `shapely`. There are some performance differences for some models due to the backends' different logics to handle invalid polygons (more info [here](https://github.com/open-mmlab/mmocr/issues/465)). **New evaluation result is presented in brackets** and new logs will be uploaded soon.
```

### Citation

```bibtex
@inproceedings{WangXSZWLYS19,
  author={Wenhai Wang and Enze Xie and Xiaoge Song and Yuhang Zang and Wenjia Wang and Tong Lu and Gang Yu and Chunhua Shen},
  title={Efficient and Accurate Arbitrary-Shaped Text Detection With Pixel Aggregation Network},
  booktitle={ICCV},
  pages={8439--8448},
  year={2019}
  }
```

## PSENet

[Shape robust text detection with progressive scale expansion network](https://arxiv.org/abs/1903.12473)

<!-- [ALGORITHM] -->

### Abstract

Scene text detection has witnessed rapid progress especially with the recent development of convolutional neural networks. However, there still exists two challenges which prevent the algorithm into industry applications. On the one hand, most of the state-of-art algorithms require quadrangle bounding box which is in-accurate to locate the texts with arbitrary shape. On the other hand, two text instances which are close to each other may lead to a false detection which covers both instances. Traditionally, the segmentation-based approach can relieve the first problem but usually fail to solve the second challenge. To address these two challenges, in this paper, we propose a novel Progressive Scale Expansion Network (PSENet), which can precisely detect text instances with arbitrary shapes. More specifically, PSENet generates the different scale of kernels for each text instance, and gradually expands the minimal scale kernel to the text instance with the complete shape. Due to the fact that there are large geometrical margins among the minimal scale kernels, our method is effective to split the close text instances, making it easier to use segmentation-based methods to detect arbitrary-shaped text instances. Extensive experiments on CTW1500, Total-Text, ICDAR 2015 and ICDAR 2017 MLT validate the effectiveness of PSENet. Notably, on CTW1500, a dataset full of long curve texts, PSENet achieves a F-measure of 74.3% at 27 FPS, and our best F-measure (82.2%) outperforms state-of-art algorithms by 6.6%. The code will be released in the future.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142795864-9b455b10-8a19-45bb-aeaf-4b733f341afc.png"/>
</div>

### Results and models

#### CTW1500

|                      Method                       | Backbone | Extra Data | Training set  |   Test set   | ##epochs | Test size |    Recall     |   Precision   |     Hmean     |                       Download                       |
| :-----------------------------------------------: | :------: | :--------: | :-----------: | :----------: | :------: | :-------: | :-----------: | :-----------: | :-----------: | :--------------------------------------------------: |
| [PSENet-4s](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/psenet/psenet_resnet50_fpnf_600e_ctw1500.py) | ResNet50 |     -      | CTW1500 Train | CTW1500 Test |   600    |   1280    | 0.728 (0.717) | 0.849 (0.852) | 0.784 (0.779) | [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/psenet/20210401_215421.log.json) |

#### ICDAR2015

|                   Method                   | Backbone |                   Extra Data                    | Training set | Test set  | ##epochs | Test size | Recall | Precision | Hmean |                   Download                    |
| :----------------------------------------: | :------: | :---------------------------------------------: | :----------: | :-------: | :------: | :-------: | :----: | :-------: | :---: | :-------------------------------------------: |
| [PSENet-4s](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/psenet/psenet_resnet50_fpnf_600e_icdar2015.py) | ResNet50 |                        -                        |  IC15 Train  | IC15 Test |   600    |   2240    | 0.766  |   0.840   | 0.806 | [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015-c6131f0d.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/psenet/20210331_214145.log.json) |
| [PSENet-4s](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/psenet/psenet_resnet50_fpnf_600e_icdar2015.py) | ResNet50 | pretrain on IC17 MLT [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2017_as_pretrain-3bd6056c.pth) |  IC15 Train  | IC15 Test |   600    |   2240    | 0.834  |   0.861   | 0.847 | [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth) \| [log](<>) |

### Citation

```bibtex
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```

## Textsnake

[TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes](https://arxiv.org/abs/1807.01544)

<!-- [ALGORITHM] -->

### Abstract

Driven by deep neural networks and large scale datasets, scene text detection methods have progressed substantially over the past years, continuously refreshing the performance records on various standard benchmarks. However, limited by the representations (axis-aligned rectangles, rotated rectangles or quadrangles) adopted to describe text, existing methods may fall short when dealing with much more free-form text instances, such as curved text, which are actually very common in real-world scenarios. To tackle this problem, we propose a more flexible representation for scene text, termed as TextSnake, which is able to effectively represent text instances in horizontal, oriented and curved forms. In TextSnake, a text instance is described as a sequence of ordered, overlapping disks centered at symmetric axes, each of which is associated with potentially variable radius and orientation. Such geometry attributes are estimated via a Fully Convolutional Network (FCN) model. In experiments, the text detector based on TextSnake achieves state-of-the-art or comparable performance on Total-Text and SCUT-CTW1500, the two newly published benchmarks with special emphasis on curved text in natural images, as well as the widely-used datasets ICDAR 2015 and MSRA-TD500. Specifically, TextSnake outperforms the baseline on Total-Text by more than 40% in F-measure.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142795949-2525ead4-865b-4762-baaa-e977cfd6ac66.png"/>
</div>

### Results and models

#### CTW1500

|                            Method                            | Pretrained Model | Training set  |   Test set   | ##epochs | Test size | Recall | Precision | Hmean |                            Download                            |
| :----------------------------------------------------------: | :--------------: | :-----------: | :----------: | :------: | :-------: | :----: | :-------: | :---: | :------------------------------------------------------------: |
| [TextSnake](https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/textsnake/textsnake_resnet50_fpn-unet_1200e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |   1200   |    736    | 0.795  |   0.840   | 0.817 | [model](https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth) \| [log](<>) |

### Citation

```bibtex
@article{long2018textsnake,
  title={TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes},
  author={Long, Shangbang and Ruan, Jiaqiang and Zhang, Wenjie and He, Xin and Wu, Wenhao and Yao, Cong},
  booktitle={ECCV},
  pages={20-36},
  year={2018}
}
```
