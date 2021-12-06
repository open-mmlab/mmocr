# SATRN

## Abstract

<!-- [ABSTRACT] -->
Scene text recognition (STR) is the task of recognizing character sequences in natural scenes. While there have been great advances in STR methods, current methods still fail to recognize texts in arbitrary shapes, such as heavily curved or rotated texts, which are abundant in daily life (e.g. restaurant signs, product labels, company logos, etc). This paper introduces a novel architecture to recognizing texts of arbitrary shapes, named Self-Attention Text Recognition Network (SATRN), which is inspired by the Transformer. SATRN utilizes the self-attention mechanism to describe two-dimensional (2D) spatial dependencies of characters in a scene text image. Exploiting the full-graph propagation of self-attention, SATRN can recognize texts with arbitrary arrangements and large inter-character spacing. As a result, SATRN outperforms existing STR models by a large margin of 5.7 pp on average in "irregular text" benchmarks. We provide empirical analyses that illustrate the inner mechanisms and the extent to which the model is applicable (e.g. rotated and multi-line text). We will open-source the code.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142798828-cc4ded5d-3fb8-478c-9f3e-74edbcf41982.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

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
| [Satrn](/configs/textrecog/satrn/satrn_academic.py) |  96.1  |     93.5     | 95.7 |     | 84.1 |      88.5      | 90.3 |      [model](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_academic_20211009-cb8b1580.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/satrn/20210809_093244.log.json)      |
| [Satrn_small](/configs/textrecog/satrn/satrn_small.py) |  94.7  |     91.3     | 95.4 |     | 81.9 |      85.9      | 86.5 |      [model](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_small_20211009-2cf13355.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/satrn/20210811_053047.log.json)      |
