# MAERec

> [Revisiting Scene Text Recognition: A Data Perspective](https://arxiv.org/abs/2307.08723)

<!-- [ALGORITHM] -->

## Abstract

This paper aims to re-assess scene text recognition (STR) from a data-oriented perspective. We begin by revisiting the six commonly used benchmarks in STR and observe a trend of performance saturation, whereby only 2.91% of the benchmark images cannot be accurately recognized by an ensemble of 13 representative models. While these results are impressive and suggest that STR could be considered solved, however, we argue that this is primarily due to the less challenging nature of the common benchmarks, thus concealing the underlying issues that STR faces. To this end, we consolidate a large-scale real STR dataset, namely Union14M, which comprises 4 million labeled images and 10 million unlabeled images, to assess the performance of STR models in more complex real-world scenarios. Our experiments demonstrate that the 13 models can only achieve an average accuracy of 66.53% on the 4 million labeled images, indicating that STR still faces numerous challenges in the real world. By analyzing the error patterns of the 13 models, we identify seven open challenges in STR and develop a challenge-driven benchmark consisting of eight distinct subsets to facilitate further progress in the field. Our exploration demonstrates that STR is far from being solved and leveraging data may be a promising solution. In this regard, we find that utilizing the 10 million unlabeled images through self-supervised pre-training can significantly improve the robustness of STR model in real-world scenarios and leads to state-of-the-art performance.

<div align=center>
<img src="https://github.com/open-mmlab/mmocr/assets/65173622/708dd6b2-b915-4d6f-b0e5-78051791dd53">
</div>

## Dataset

### Train Dataset

| trainset | instance_num | repeat_num | source |
| :------: | :----------: | :--------: | :----: |
| [Union14M](https://github.com/Mountchicken/Union14M#34-download) |   3230742    |     1      |  real  |

### Test Dataset
- On six common benchmarks

    | testset | instance_num |   type    |
    | :-----: | :----------: | :-------: |
    | IIIT5K  |     3000     |  regular  |
    |   SVT   |     647      |  regular  |
    |  IC13   |     1015     |  regular  |
    |  IC15   |     2077     | irregular |
    |  SVTP   |     645      | irregular |
    |  CT80   |     288      | irregular |

- On Union14M-Benchmark

    |    testset     | instance_num |         type         |
    | :------------: | :----------: | :------------------: |
    |    Artistic    |     900      |  Unsolved Challenge  |
    |     Curve      |     2426     |  Unsolved Challenge  |
    | Multi-Oriented |     1369     |  Unsolved Challenge  |
    |  Contextless   |     779      | Additional Challenge |
    |  Multi-Words   |     829      | Additional Challenge |
    |    Salient     |     1585     | Additional Challenge |
    |   Incomplete   |     1495     | Additional Challenge |
    |    General     |   400,000    |          -           |


## Results and Models

- Evaluated on six common benchmarks

    |                          Methods                          |   Backbone    |        | Regular Text |           |       |           | Irregular Text |        |                                                                                                                                download                                                                                                                                |
    | :-------------------------------------------------------: | :-----------: | :----: | :----------: | :-------: | :---: | :-------: | :------------: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
    |                                                           |               | IIIT5K |     SVT      | IC13-1015 |       | IC15-2077 |      SVTP      |  CT80  |                                                                                                                                                                                                                                                                        |
    | [MAERec-S](configs/textrecog/maerec/maerec_s_union14m.py) | [ViT-Small (Pretrained on Union14M-U)](https://github.com/Mountchicken/Union14M#51-pre-training) | 98.0 |    97.6    |  96.8   |       |  87.1   |     93.2     | 97.9 | [model](https://download.openmmlab.com/mmocr/textrecog/mae/mae_union14m/maerec_s_union14m-a9a157e5.pth) |
    | [MAERec-B](configs/textrecog/maerec/maerec_b_union14m.py) | [ViT-Base (Pretrained on Union14M-U)](https://github.com/Mountchicken/Union14M#51-pre-training)| 98.5 |    98.1    |  97.8   |       |  89.5   |     94.4     | 98.6 |       [model](https://download.openmmlab.com/mmocr/textrecog/mae/mae_union14m/maerec_b_union14m-4b98d1b4.pth)                                                                                                                                                                                                                                                                  |

- Evaluated on Union14M-Benchmark

    |Methods|Backbone||Unsolved Challenges|||||Additional Challenges||General|download|
    |----|----|----|----|----|----|----|----|----|----|----|----|
    |||Curve|Multi-Oriented|Artistic|Contextless||Salient|Multi-Words|Incomplete|General|
    |[MAERec-S](configs/textrecog/maerec/maerec_s_union14m.py)|[ViT-Small (Pretrained on Union14M-U)](https://github.com/Mountchicken/Union14M#51-pre-training)|81.4|71.4|72.0|82.0||78.5|82.4|2.7|82.5|[model](https://download.openmmlab.com/mmocr/textrecog/mae/mae_union14m/maerec_s_union14m-a9a157e5.pth)|
    |[MAERec-B](configs/textrecog/maerec/maerec_b_union14m.py)|[ViT-Base (Pretrained on Union14M-U)](https://github.com/Mountchicken/Union14M#51-pre-training)|88.8|83.9|80.0|85.5||84.9|87.5|2.6|85.8|[model](https://download.openmmlab.com/mmocr/textrecog/mae/mae_union14m/maerec_b_union14m-4b98d1b4.pth)         |

## Citation

```bibtex
@misc{jiang2023revisiting,
      title={Revisiting Scene Text Recognition: A Data Perspective}, 
      author={Qing Jiang and Jiapeng Wang and Dezhi Peng and Chongyu Liu and Lianwen Jin},
      year={2023},
      eprint={2307.08723},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
