# Textsnake

## Introduction

[ALGORITHM]

```bibtex
@article{long2018textsnake,
  title={TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes},
  author={Long, Shangbang and Ruan, Jiaqiang and Zhang, Wenjie and He, Xin and Wu, Wenhao and Yao, Cong},
  booktitle={ECCV},
  pages={20-36},
  year={2018}
}
```

## Results and models

### CTW1500

|                                    Method                                      | Pretrained Model | Training set  |   Test set   | #epochs | Test size | Recall | Precision | Hmean |       Download        |
| :----------------------------------------------------------------------------: | :--------------: | :-----------: | :----------: | :-----: | :-------: | :----: | :-------: | :---: | :-------------------: |
| [TextSnake](/configs/textdet/textsnake/textsnake_r50_fpn_unet_600e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |  1200   |    736    | 0.795  |   0.840   | 0.817 | [model](https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth) &#124; [config](https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py) |
