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

|                            Method                            | Pretrained Model | Training set  |   Test set   | #epochs | Test size | Recall | Precision | Hmean |                            Download                             |
| :----------------------------------------------------------: | :--------------: | :-----------: | :----------: | :-----: | :-------: | :----: | :-------: | :---: | :-------------------------------------------------------------: |
| [MaskRCNN](/configs/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |   160   |   1600    | 0.753  |   0.712   | 0.732 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.log.json) |

### ICDAR2015

|                           Method                           | Pretrained Model |  Training set   |    Test set    | #epochs | Test size | Recall | Precision | Hmean |                           Download                            |
| :--------------------------------------------------------: | :--------------: | :-------------: | :------------: | :-----: | :-------: | :----: | :-------: | :---: | :-----------------------------------------------------------: |
| [MaskRCNN](/configs/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py) |     ImageNet     | ICDAR2015 Train | ICDAR2015 Test |   160   |   1920    | 0.783  |   0.872   | 0.825 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.log.json) |

### ICDAR2017

|                           Method                            | Pretrained Model |  Training set   |   Test set    | #epochs | Test size | Recall | Precision | Hmean |                           Download                            |
| :---------------------------------------------------------: | :--------------: | :-------------: | :-----------: | :-----: | :-------: | :----: | :-------: | :---: | :-----------------------------------------------------------: |
| [MaskRCNN](/configs/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py) |     ImageNet     | ICDAR2017 Train | ICDAR2017 Val |   160   |   1600    | 0.754  |   0.827   | 0.789 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.log.json) |

```{note}
We tuned parameters with the techniques in [Pyramid Mask Text Detector](https://arxiv.org/abs/1903.11800)
```

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
