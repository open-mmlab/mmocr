# SATRN

## Introduction

[ALGORITHM]

```bibtex
@article{junyeop2019recognizing,
  title={On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention},
  author={Junyeop Lee, Sungrae Park, Jeonghun Baek, Seong Joon Oh, Seonghyeon Kim, Hwalsuk Lee},
  year={2019}
}
```

## Dataset

### Train Dataset

| trainset  | instance_num | repeat_num | source |
| :-------: | :----------: | :--------: | :----: |
| SynthText |   7266686    |     1      | synth  |
|  Syn90k   |   8919273    |     1      | synth  |

### Test Dataset

| testset | instance_num |   type    |
| :-----: | :----------: | :-------: |
| IIIT5K  |     3000     |  regular  |
|   SVT   |     647      |  regular  |
|  IC13   |     1015     |  regular  |
|  IC15   |     2077     | irregular |
|  SVTP   |     645      | irregular |
|  CT80   |     288      | irregular |

## Results and Models

|                             Methods                             |        | Regular Text |      |     |      | Irregular Text |      |                                                                                               download                                                                                                |
| :-------------------------------------------------------------: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                 | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |
| [Satrn](/configs/textrecog/satrn/satrn_academic.py) |  96.1  |     93.5     | 95.7 |     | 84.1 |      88.5      | 90.3 |      [model](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_academic_20210809-59c8c92d.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/satrn/20210809_093244.log.json)      |
| [Satrn_small](/configs/textrecog/satrn/satrn_small.py) |  94.7  |     91.3     | 95.4 |     | 81.9 |      85.9      | 86.5 |      [model](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_academic_20210809-59c8c92d.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/satrn/20210811_053047.log.json)      |
