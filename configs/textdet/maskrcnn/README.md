# Mask R-CNN

> [Mask R-CNN](https://arxiv.org/abs/1703.06870)

<!-- [ALGORITHM] -->

## Abstract

We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142795605-dfdd5f69-e9cd-4b69-9c6b-6d8bded18e89.png"/>
</div>

## Results and models

### CTW1500

|                 Method                  |                 BackBone                  | Pretrained Model | Training set  |   Test set   | #epochs | Test size | Precision | Recall | Hmean  |                  Download                  |
| :-------------------------------------: | :---------------------------------------: | :--------------: | :-----------: | :----------: | :-----: | :-------: | :-------: | :----: | :----: | :----------------------------------------: |
| [MaskRCNN](/configs/textdet/maskrcnn/mask-rcnn_resnet50_fpn_160e_ctw1500.py) |                     -                     |        -         | CTW1500 Train | CTW1500 Test |   160   |   1600    |  0.7165   | 0.7776 | 0.7458 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50_fpn_160e_ctw1500/mask-rcnn_resnet50_fpn_160e_ctw1500_20220826_154755-ce68ee8e.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50_fpn_160e_ctw1500/20220826_154755.log) |
| [MaskRCNN_r50-oclip](/configs/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_ctw1500.py) | [ResNet50-oCLIP](https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth) |        -         | CTW1500 Train | CTW1500 Test |   160   |   1600    |   0.753   | 0.7593 | 0.7562 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_ctw1500/mask-rcnn_resnet50-oclip_fpn_160e_ctw1500_20221101_154448-6e9e991c.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_ctw1500/20221101_154448.log) |

### ICDAR2015

|                 Method                 |                 BackBone                 | Pretrained Model |  Training set   |    Test set    | #epochs | Test size | Precision | Recall | Hmean  |                 Download                 |
| :------------------------------------: | :--------------------------------------: | :--------------: | :-------------: | :------------: | :-----: | :-------: | :-------: | :----: | :----: | :--------------------------------------: |
| [MaskRCNN](/configs/textdet/maskrcnn/mask-rcnn_resnet50_fpn_160e_icdar2015.py) |                 ResNet50                 |        -         | ICDAR2015 Train | ICDAR2015 Test |   160   |   1920    |  0.8644   | 0.7766 | 0.8182 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50_fpn_160e_icdar2015/mask-rcnn_resnet50_fpn_160e_icdar2015_20220826_154808-ff5c30bf.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50_fpn_160e_icdar2015/20220826_154808.log) |
| [MaskRCNN_r50-oclip](/configs/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_icdar2015.py) | [ResNet50-oCLIP](https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth) |        -         | ICDAR2015 Train | ICDAR2015 Test |   160   |   1920    |  0.8695   | 0.8339 | 0.8513 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_icdar2015/mask-rcnn_resnet50-oclip_fpn_160e_icdar2015_20221101_131357-a19f7802.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_icdar2015/20221101_131357.log) |

## Citation

```bibtex
@INPROCEEDINGS{8237584,
  author={K. {He} and G. {Gkioxari} and P. {Dollár} and R. {Girshick}},
  booktitle={2017 IEEE International Conference on Computer Vision (ICCV)},
  title={Mask R-CNN},
  year={2017},
  pages={2980-2988},
  doi={10.1109/ICCV.2017.322}}
```
