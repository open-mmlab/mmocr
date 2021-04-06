# Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network

## Introduction

[ALGORITHM]

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
| [PANet](/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |   600   |    640    | 0.790  |   0.838   | 0.813 | [model](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.log.json) |

### ICDAR2015

|                                Method                                | Pretrained Model |  Training set   |    Test set    | #epochs | Test size | Recall | Precision | Hmean |                                                                                                                       Download                                                                                                                        |
| :------------------------------------------------------------------: | :--------------: | :-------------: | :------------: | :-----: | :-------: | :----: | :-------: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PANet](/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |     ImageNet     | ICDAR2015 Train | ICDAR2015 Test |   600   |    736    | 0.734  |   0.856   | 0.791 | [model](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.log.json) |

### ICDAR2017

|                                Method                                | Pretrained Model |  Training set   |   Test set    | #epochs | Test size | Recall | Precision | Hmean |                                                                                                                       Download                                                                                                                        |
| :------------------------------------------------------------------: | :--------------: | :-------------: | :-----------: | :-----: | :-------: | :----: | :-------: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PANet](/configs/textdet/panet/panet_r50_fpem_ffm_600e_icdar2017.py) |     ImageNet     | ICDAR2017 Train | ICDAR2017 Val |   600   |    800    | 0.604  |   0.812   | 0.693 | [model](https://download.openmmlab.com/mmocr/textdet/panet/panet_r50_fpem_ffm_sbn_600e_icdar2017_20210219-b4877a4f.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/panet/panet_r50_fpem_ffm_sbn_600e_icdar2017_20210219-b4877a4f.log.json) |
