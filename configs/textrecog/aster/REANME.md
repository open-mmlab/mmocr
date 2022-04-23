# ASTER
>[ASTER: An Attentional Scene Text Recognizer with Flexible Rectification](https://ieeexplore.ieee.org/abstract/document/8395027)

<!-- [ALGORITHM] -->

## Abstract
A challenging aspect of scene text recognition is to handle text with distortions or irregular layout. In particular, perspective text and curved text are common in natural scenes and are difficult to recognize. In this work, we introduce ASTER, an end-to-end neural network model that comprises a rectification network and a recognition network. The rectification network adaptively transforms an input image into a new one, rectifying the text in it. It is powered by a flexible Thin-Plate Spline transformation which handles a variety of text irregularities and is trained without human annotations. The recognition network is an attentional sequence-to-sequence model that predicts a character sequence directly from the rectified image. The whole model is trained end to end, requiring only images and their groundtruth text. Through extensive experiments, we verify the effectiveness of the rectification and demonstrate the state-of-the-art recognition performance of ASTER. Furthermore, we demonstrate that ASTER is a powerful component in end-to-end recognition systems, for its ability to enhance the detector.

<div align=center>
<img src="" >
</div>

## Dataset

### Train Dataset

| trainset  | instance_num | repeat_num | source |
| :-------: | :----------: | :--------: | :----: |
| SynthText |   7266686    |     1      | synth  |
|  Syn90k   |   8919273    |     1      | synth  |

### Test Dataset
| testset | instance_num |   note    |
| :-----: | :----------: | :-------: |
| IIIT5K  |     3000     |  regular  |
|   SVT   |     647      |  regular  |
|  IC13   |     1015     |  regular  |
|  IC15   |     2077     | irregular |
|  SVTP   |     645      | irregular |
|  CT80   |     288      | irregular |

### Results and Models


## Citation

```bibtex
@article{bshi2018aster,
  author    = {Baoguang Shi and
               Mingkun Yang and
               Xinggang Wang and
               Pengyuan Lyu and
               Cong Yao and
               Xiang Bai},
  title     = {ASTER: An Attentional Scene Text Recognizer with Flexible Rectification},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume    = {41},
  number    = {9},
  pages     = {2035--2048},
  year      = {2019},
}

@inproceedings{ShiWLYB16,
  author    = {Baoguang Shi and
               Xinggang Wang and
               Pengyuan Lyu and
               Cong Yao and
               Xiang Bai},
  title     = {Robust Scene Text Recognition with Automatic Rectification},
  booktitle = {2016 {IEEE} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2016, Las Vegas, NV, USA, June 27-30, 2016},
  pages     = {4168--4176},
  year      = {2016}
}
```
