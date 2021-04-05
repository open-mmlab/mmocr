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

## Results and models

| methods |        | Regular Text |      |     |      | Irregular Text |      |       download       |
| :-----: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :------------------: |
| methods | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |
|  CRNN   |  80.5  |     81.5     | 86.5 |     |  -   |       -        |  -   | [model]() \| [log]() |
