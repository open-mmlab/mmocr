# PSENet

## Introduction

[ALGORITHM]

```bibtex
@article{li2018shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Li, Xiang and Wang, Wenhai and Hou, Wenbo and Liu, Ruo-Ze and Lu, Tong and Yang, Jian},
  journal={arXiv preprint arXiv:1806.02559},
  year={2018}
}
```

## Results and models

### CTW1500

|Method | Backbone|Extra Data | Training set | Test set | #epochs | Test size|Recall|Precision|Hmean|Download|
|:------:| :------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|[PSENet-4s](/configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py) |ResNet50 |-|CTW1500 Train|CTW1500 Test|600|1280|0.771|0.793|0.782|[model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_ctw1500-ac38d587.pth) \| [config](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py)|

### ICDAR2015

|Method | Backbone| Extra Data | Training set | Test set | #epochs | Test size|Recall|Precision|Hmean|Download|
|:------:| :------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|[PSENet-4s](/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py) |ResNet50 |-|IC15 Train|IC15 Test|600|2240|0.784|0.831|0.807|[model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015-c6131f0d.pth) &#124; [config](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py)  &#124; [log](https://download.openmmlab.com/mmocr/textdet/psenet/20210331_214145.log.json)|
|[PSENet-4s](/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py) |ResNet50 |pretrain on IC17 MLT [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2017_as_pretrain-0af6d62c.pth)|IC15 Train|IC15 Test|600|2240|0.834|0.861|0.847|[model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-ac477383.pth) &#124; [config](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py) |
