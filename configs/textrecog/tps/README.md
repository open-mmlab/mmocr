# CRNN with TPS based STN

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

[PREPROCESSOR]

```bibtex
@article{shi2016robust,
  title={Robust Scene Text Recognition with Automatic Rectification},
  author={Shi, Baoguang and Wang, Xinggang and Lyu, Pengyuan and Yao,
  Cong and Bai, Xiang},
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
|                                                   | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |
| [CRNN-STN](/configs/textrecog/tps/crnn_tps_academic_dataset.py) |  80.8  |     81.3     | 85.0 |     |  59.6   |       68.1        |  53.8   | [model](https://download.openmmlab.com/mmocr/textrecog/tps/crnn_tps_academic_dataset_20210510-d221a905.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/tps/20210510_204353.log.json) |
