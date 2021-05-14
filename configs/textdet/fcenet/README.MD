# Fourier Contour Embedding for Arbitrary-Shaped Text Detection

## Introduction

[ALGORITHM]

```bibtex
@InProceedings{zhu2021fourier,
      title={Fourier Contour Embedding for Arbitrary-Shaped Text Detection}, 
      author={Yiqin Zhu and Jianyong Chen and Lingyu Liang and Zhanghui Kuang and Lianwen Jin and Wayne Zhang},
      year={2021},
      booktitle = {CVPR}
      }
```

## Results and models

### CTW1500

|                                     Method                             | Pretrained Model | Training set  |   Test set   | #epochs | Test size | Recall | Precision | Hmean |                                                                                        Download                                                                                                    |
| :--------------------------------------------------------------------: | :--------------: | :-----------: | :----------: | :-----: | :-------: | :----: | :-------: | :---: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [FCENet](/configs/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |  1500   |(736, 1080)| 0.828  |   0.875   | 0.851 | [model](https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500-05d740bb.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/fcenet/20210511_181328.log.json) |
