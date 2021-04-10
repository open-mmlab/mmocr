# RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition

## Introduction

[ALGORITHM]

```bibtex
@inproceedings{yue2020robustscanner,
  title={RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition},
  author={Yue, Xiaoyu and Kuang, Zhanghui and Lin, Chenhao and Sun, Hongbin and Zhang, Wayne},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

## Dataset

### Train Dataset

|  trainset  | instance_num | repeat_num |          source          |
| :--------: | :----------: | :--------: | :----------------------: |
| icdar_2011 |     3567     |     20     |           real           |
| icdar_2013 |     848      |     20     |           real           |
| icdar2015  |     4468     |     20     |           real           |
| coco_text  |    42142     |     20     |           real           |
|   IIIT5K   |     2000     |     20     |           real           |
| SynthText  |   2400000    |     1      |          synth           |
|  SynthAdd  |   1216889    |     1      | synth, 1.6m in [[1]](#1) |
|   Syn90k   |   2400000    |     1      |          synth           |

### Test Dataset

| testset | instance_num |            type             |
| :-----: | :----------: | :-------------------------: |
| IIIT5K  |     3000     |           regular           |
|   SVT   |     647      |           regular           |
|  IC13   |     1015     |           regular           |
|  IC15   |     2077     |          irregular          |
|  SVTP   |     645      | irregular, 639 in [[1]](#1) |
|  CT80   |     288      |          irregular          |

## Results and Models

|                                     Methods                                     | GPUs |        | Regular Text |      |     |      | Irregular Text |      |                                                                                                   download                                                                                                    |
| :-----------------------------------------------------------------------------: | :--: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                 |      | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |
| [RobustScanner](configs/textrecog/robust_scanner/robustscanner_r31_academic.py) |  16  |  95.1  |     89.2     | 93.1 |     | 77.8 |      80.3      | 90.3 | [model](https://download.openmmlab.com/mmocr/textrecog/robustscanner/robustscanner_r31_academic-5f05874f.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/robustscanner/20210401_170932.log.json) |

## References

<a id="1">[1]</a> Li, Hui and Wang, Peng and Shen, Chunhua and Zhang, Guyu. Show, attend and read: A simple and strong baseline for irregular text recognition. In AAAI 2019.
