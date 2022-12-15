# ASTER

> [ASTER: An Attentional Scene Text Recognizer with Flexible Rectification](https://ieeexplore.ieee.org/abstract/document/8395027/)

<!-- [ALGORITHM] -->

## Abstract

A challenging aspect of scene text recognition is to handle text with distortions or irregular layout. In particular, perspective text and curved text are common in natural scenes and are difficult to recognize. In this work, we introduce ASTER, an end-to-end neural network model that comprises a rectification network and a recognition network. The rectification network adaptively transforms an input image into a new one, rectifying the text in it. It is powered by a flexible Thin-Plate Spline transformation which handles a variety of text irregularities and is trained without human annotations. The recognition network is an attentional sequence-to-sequence model that predicts a character sequence directly from the rectified image. The whole model is trained end to end, requiring only images and their groundtruth text. Through extensive experiments, we verify the effectiveness of the rectification and demonstrate the state-of-the-art recognition performance of ASTER. Furthermore, we demonstrate that ASTER is a powerful component in end-to-end recognition systems, for its ability to enhance the detector.

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/207841597-fcd596cf-20eb-42db-9108-21e586dd9109.png"/>
</div>

## Dataset

### Train Dataset

| trainset  | instance_num | repeat_num |     note     |
| :-------: | :----------: | :--------: | :----------: |
|  Syn90k   |   8919273    |     1      |    synth     |
| SynthText |   7239272    |     1      | alphanumeric |

### Test Dataset

| testset | instance_num |   note    |
| :-----: | :----------: | :-------: |
| IIIT5K  |     3000     |  regular  |
|   SVT   |     647      |  regular  |
|  IC13   |     1015     |  regular  |
|  IC15   |     2077     | irregular |
|  SVTP   |     645      | irregular |
|  CT80   |     288      | irregular |

## Results and models

|                       Methods                       | Backbone |        | Regular Text |           |     |           | Irregular Text |        |                                      download                                      |
| :-------------------------------------------------: | :------: | :----: | :----------: | :-------: | :-: | :-------: | :------------: | :----: | :--------------------------------------------------------------------------------: |
|                                                     |          | IIIT5K |     SVT      | IC13-1015 |     | IC15-2077 |      SVTP      |  CT80  |                                                                                    |
| [ASTER](/configs/textrecog/aster/aster_6e_st_mj.py) | ResNet45 | 0.9357 |    0.8949    |  0.9281   |     |  0.7665   |     0.8062     | 0.8507 | [model](https://download.openmmlab.com/mmocr/textrecog/aster/aster_resnet45_6e_st_mj/aster_resnet45_6e_st_mj-cc56eca4.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/aster/aster_resnet45_6e_st_mj/20221214_232605.log) |

## Citation

```bibtex
@article{shi2018aster,
  title={Aster: An attentional scene text recognizer with flexible rectification},
  author={Shi, Baoguang and Yang, Mingkun and Wang, Xinggang and Lyu, Pengyuan and Yao, Cong and Bai, Xiang},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={41},
  number={9},
  pages={2035--2048},
  year={2018},
  publisher={IEEE}
}
```
