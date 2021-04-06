# Mask R-CNN

## Introduction

[ALGORITHM]

```bibtex
@article{pmtd,
  author={Jingchao Liu and Xuebo Liu and Jie Sheng and Ding Liang and Xin Li and Qingjie Liu},
  title={Pyramid Mask Text Detector},
  journal={CoRR},
  volume={abs/1903.11800},
  year={2019}
}
```

## Results and models

### CTW1500

|                                 Method                                  | Pretrained Model | Training set  |   Test set   | #epochs | Test size | Recall | Precision | Hmean |                                                                                                                   Download                                                                                                                    |
| :---------------------------------------------------------------------: | :--------------: | :-----------: | :----------: | :-----: | :-------: | :----: | :-------: | :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MaskRCNN](/configs/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |   160   |   1600    | 0.753  |   0.712   | 0.732 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.log.json) |

### ICDAR2015

|                                  Method                                   | Pretrained Model |  Training set   |    Test set    | #epochs | Test size | Recall | Precision | Hmean |                                                                                                                     Download                                                                                                                      |
| :-----------------------------------------------------------------------: | :--------------: | :-------------: | :------------: | :-----: | :-------: | :----: | :-------: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MaskRCNN](/configs/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py) |     ImageNet     | ICDAR2015 Train | ICDAR2015 Test |   160   |   1920    | 0.783  |   0.872   | 0.825 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.log.json) |

### ICDAR2017

|                                  Method                                   | Pretrained Model |  Training set   |   Test set    | #epochs | Test size | Recall | Precision | Hmean |                                                                                                                     Download                                                                                                                      |
| :-----------------------------------------------------------------------: | :--------------: | :-------------: | :-----------: | :-----: | :-------: | :----: | :-------: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MaskRCNN](/configs/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py) |     ImageNet     | ICDAR2017 Train | ICDAR2017 Val |   160   |   1600    | 0.754  |   0.827   | 0.789 | [model](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.log.json) |
