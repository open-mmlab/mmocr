# An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition

## Introduction

[ALGORITHM]

```bibtex
@article{shi2016end,
  title={An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition},
  author={Shi, Baoguang and Bai, Xiang and Yao, Cong},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2016}
}
```

## Results and Models

### Train Dataset

| trainset | instance_num | repeat_num | note  |
| :------: | :----------: | :--------: | :---: |
|  Syn90k  |   8919273    |     1      | synth |

### Test Dataset

| testset | instance_num |  note   |
| :-----: | :----------: | :-----: |
| IIIT5K  |     3000     | regular |
|   SVT   |     647      | regular |
|  IC13   |     1015     | regular |
|  IC15   |     2077     |irregular|
|  SVTP   |     645      |irregular|
|  CT80   |     288      |irregular|

## Results and models

|                         methods                          |        | Regular Text |      |     |      | Irregular Text |      |                                                                                    download                                                                                    |
| :------------------------------------------------------: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                         methods                          | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |
| [CRNN](/configs/textrecog/crnn/crnn_academic_dataset.py) |  80.5  |     81.5     | 86.5 |     |  54.1   |       59.1        |  55.6   | [model](https://download.openmmlab.com/mmocr/textrecog/crnn/crnn_academic-a723a1c5.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/crnn/20210326_111035.log.json) |
