# Real-time Scene Text Detection with Differentiable Binarization

## Introduction

[ALGORITHM]

```bibtex
@article{Liao_Wan_Yao_Chen_Bai_2020,
    title={Real-Time Scene Text Detection with Differentiable Binarization},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
    year={2020},
    pages={11474-11481}}
```

## Results and models

### ICDAR2015

|                                 Method                                 |                                                      Pretrained Model                                                      |  Training set   |    Test set    | #epochs | Test size | Recall | Precision | Hmean |                                                                                                                           Download                                                                                                                            |
| :--------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------: | :-------------: | :------------: | :-----: | :-------: | :----: | :-------: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [DBNet_r18](/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py) |     ImageNet     | ICDAR2015 Train | ICDAR2015 Test |  1200   |    736    | 0.731  |   0.871   | 0.795 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.log.json) |
| [DBNet_r50dcn](/configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py) | [Synthtext](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-aa96e477.pth) | ICDAR2015 Train | ICDAR2015 Test |  1200   |   1024    | 0.814  |   0.868   | 0.840 | [model](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.log.json) |
