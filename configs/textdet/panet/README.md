# Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network

## Abstract

<!-- [ABSTRACT] -->
Scene text detection, an important step of scene text reading systems, has witnessed rapid development with convolutional neural networks. Nonetheless, two main challenges still exist and hamper its deployment to real-world applications. The first problem is the trade-off between speed and accuracy. The second one is to model the arbitrary-shaped text instance. Recently, some methods have been proposed to tackle arbitrary-shaped text detection, but they rarely take the speed of the entire pipeline into consideration, which may fall short in practical this http URL this paper, we propose an efficient and accurate arbitrary-shaped text detector, termed Pixel Aggregation Network (PAN), which is equipped with a low computational-cost segmentation head and a learnable post-processing. More specifically, the segmentation head is made up of Feature Pyramid Enhancement Module (FPEM) and Feature Fusion Module (FFM). FPEM is a cascadable U-shaped module, which can introduce multi-level information to guide the better segmentation. FFM can gather the features given by the FPEMs of different depths into a final feature for segmentation. The learnable post-processing is implemented by Pixel Aggregation (PA), which can precisely aggregate text pixels by predicted similarity vectors. Experiments on several standard benchmarks validate the superiority of the proposed PAN. It is worth noting that our method can achieve a competitive F-measure of 79.9% at 84.2 FPS on CTW1500.


<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142795741-0e1ea962-1596-47c2-8671-27bbe87d0df8.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{WangXSZWLYS19,
  author={Wenhai Wang and Enze Xie and Xiaoge Song and Yuhang Zang and Wenjia Wang and Tong Lu and Gang Yu and Chunhua Shen},
  title={Efficient and Accurate Arbitrary-Shaped Text Detection With Pixel Aggregation Network},
  booktitle={ICCV},
  pages={8439--8448},
  year={2019}
  }
```

## Results and models

### CTW1500

|                               Method                               | Pretrained Model | Training set  |   Test set   | #epochs | Test size | Recall | Precision | Hmean |                                                                                                                     Download                                                                                                                      |
| :----------------------------------------------------------------: | :--------------: | :-----------: | :----------: | :-----: | :-------: | :----: | :-------: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PANet](configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |   600   |    640    | 0.776 (0.717)  |   0.838 (0.835)  | 0.806 (0.801) | [model](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.log.json) |

### ICDAR2015

|                                Method                                | Pretrained Model |  Training set   |    Test set    | #epochs | Test size | Recall | Precision | Hmean |                                                                                                                       Download                                                                                                                        |
| :------------------------------------------------------------------: | :--------------: | :-------------: | :------------: | :-----: | :-------: | :----: | :-------: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PANet](configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |     ImageNet     | ICDAR2015 Train | ICDAR2015 Test |   600   |    736    | 0.734 (0.74) |   0.856 (0.86)  | 0.791 (0.795) | [model](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.log.json) |

**Note:** We've upgraded our IoU backend from `Polygon3` to `shapely`. There are some performance differences for some models due to the backends' different logics to handle invalid polygons (more info [here](https://github.com/open-mmlab/mmocr/issues/465)). **New evaluation result is presented in brackets** and new logs will be uploaded soon.
