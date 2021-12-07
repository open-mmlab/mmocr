# PSENet

## Abstract

<!-- [ABSTRACT] -->
Scene text detection has witnessed rapid progress especially with the recent development of convolutional neural networks. However, there still exists two challenges which prevent the algorithm into industry applications. On the one hand, most of the state-of-art algorithms require quadrangle bounding box which is in-accurate to locate the texts with arbitrary shape. On the other hand, two text instances which are close to each other may lead to a false detection which covers both instances. Traditionally, the segmentation-based approach can relieve the first problem but usually fail to solve the second challenge. To address these two challenges, in this paper, we propose a novel Progressive Scale Expansion Network (PSENet), which can precisely detect text instances with arbitrary shapes. More specifically, PSENet generates the different scale of kernels for each text instance, and gradually expands the minimal scale kernel to the text instance with the complete shape. Due to the fact that there are large geometrical margins among the minimal scale kernels, our method is effective to split the close text instances, making it easier to use segmentation-based methods to detect arbitrary-shaped text instances. Extensive experiments on CTW1500, Total-Text, ICDAR 2015 and ICDAR 2017 MLT validate the effectiveness of PSENet. Notably, on CTW1500, a dataset full of long curve texts, PSENet achieves a F-measure of 74.3% at 27 FPS, and our best F-measure (82.2%) outperforms state-of-art algorithms by 6.6%. The code will be released in the future.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142795864-9b455b10-8a19-45bb-aeaf-4b733f341afc.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```

## Results and models

### CTW1500

|                                Method                                | Backbone | Extra Data | Training set  |   Test set   | #epochs | Test size | Recall | Precision | Hmean |                                                                                                Download                                                                                                |
| :------------------------------------------------------------------: | :------: | :--------: | :-----------: | :----------: | :-----: | :-------: | :----: | :-------: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PSENet-4s](configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py) | ResNet50 |     -      | CTW1500 Train | CTW1500 Test |   600   |   1280    | 0.728 (0.717)  |   0.849 (0.852)  | 0.784 (0.779) | [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/psenet/20210401_215421.log.json) |

### ICDAR2015

|                                 Method                                 | Backbone |                                                                Extra Data                                                                 | Training set | Test set  | #epochs | Test size | Recall | Precision | Hmean |                                                                                            Download                                                                                             |
| :--------------------------------------------------------------------: | :------: | :---------------------------------------------------------------------------------------------------------------------------------------: | :----------: | :-------: | :-----: | :-------: | :----: | :-------: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PSENet-4s](configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py) | ResNet50 |                                                                     -                                                                     |  IC15 Train  | IC15 Test |   600   |   2240    | 0.784 (0.753)  |   0.831 (0.867)   | 0.807 (0.806) | [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015-c6131f0d.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/psenet/20210331_214145.log.json) |
| [PSENet-4s](configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py) | ResNet50 | pretrain on IC17 MLT [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2017_as_pretrain-3bd6056c.pth) |  IC15 Train  | IC15 Test |   600   |   2240    | 0.834  |   0.861   | 0.847 |                                  [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth) \| [log]()                                   |

**Note:** We've upgraded our IoU backend from `Polygon3` to `shapely`. There are some performance differences for some models due to the backends' different logics to handle invalid polygons (more info [here](https://github.com/open-mmlab/mmocr/issues/465)). **New evaluation result is presented in brackets** and new logs will be uploaded soon.
