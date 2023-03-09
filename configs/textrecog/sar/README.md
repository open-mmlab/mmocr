# SAR

> [Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/abs/1811.00751)

<!-- [ALGORITHM] -->

## Abstract

Recognizing irregular text in natural scene images is challenging due to the large variance in text appearance, such as curvature, orientation and distortion. Most existing approaches rely heavily on sophisticated model designs and/or extra fine-grained annotations, which, to some extent, increase the difficulty in algorithm implementation and data collection. In this work, we propose an easy-to-implement strong baseline for irregular scene text recognition, using off-the-shelf neural network components and only word-level annotations. It is composed of a 31-layer ResNet, an LSTM-based encoder-decoder framework and a 2-dimensional attention module. Despite its simplicity, the proposed method is robust and achieves state-of-the-art performance on both regular and irregular scene text recognition benchmarks.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142798157-ac68907f-5a8a-473f-a29f-f0532b7fdba0.png"/>
</div>

## Dataset

### Train Dataset

|  trainset  | instance_num | repeat_num |           source           |
| :--------: | :----------: | :--------: | :------------------------: |
| icdar_2011 |     3567     |     20     |            real            |
| icdar_2013 |     848      |     20     |            real            |
| icdar2015  |     4468     |     20     |            real            |
| coco_text  |    42142     |     20     |            real            |
|   IIIT5K   |     2000     |     20     |            real            |
| SynthText  |   2400000    |     1      |           synth            |
|  SynthAdd  |   1216889    |     1      | synth, 1.6m in [\[1\]](#1) |
|   Syn90k   |   2400000    |     1      |           synth            |

### Test Dataset

| testset | instance_num |             type              |
| :-----: | :----------: | :---------------------------: |
| IIIT5K  |     3000     |            regular            |
|   SVT   |     647      |            regular            |
|  IC13   |     1015     |            regular            |
|  IC15   |     2077     |           irregular           |
|  SVTP   |     645      | irregular, 639 in [\[1\]](#1) |
|  CT80   |     288      |           irregular           |

## Results and Models

|                        Methods                         |  Backbone   |       Decoder        |        | Regular Text |           |     |           | Irregular Text |        |                         download                         |
| :----------------------------------------------------: | :---------: | :------------------: | :----: | :----------: | :-------: | :-: | :-------: | :------------: | :----: | :------------------------------------------------------: |
|                                                        |             |                      | IIIT5K |     SVT      | IC13-1015 |     | IC15-2077 |      SVTP      |  CT80  |                                                          |
| [SAR](/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py) | R31-1/8-1/4 |  ParallelSARDecoder  | 0.9533 |    0.8964    |  0.9369   |     |  0.7602   |     0.8326     | 0.9062 | [model](https://download.openmmlab.com/mmocr/textrecog/sar/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real_20220915_171910-04eb4e75.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/sar/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real/20220915_171910.log) |
| [SAR-TTA](/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py) | R31-1/8-1/4 |  ParallelSARDecoder  | 0.9510 |    0.8964    |  0.9340   |     |  0.7862   |     0.8372     | 0.9132 |                                                          |
| [SAR](/configs/textrecog/sar/sar_r31_sequential_decoder_academic.py) | R31-1/8-1/4 | SequentialSARDecoder | 0.9553 |    0.9073    |  0.9409   |     |  0.7761   |     0.8093     | 0.8958 | [model](https://download.openmmlab.com/mmocr/textrecog/sar/sar_resnet31_sequential-decoder_5e_st-sub_mj-sub_sa_real/sar_resnet31_sequential-decoder_5e_st-sub_mj-sub_sa_real_20220915_185451-1fd6b1fc.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/sar/sar_resnet31_sequential-decoder_5e_st-sub_mj-sub_sa_real/20220915_185451.log) |
| [SAR-TTA](/configs/textrecog/sar/sar_r31_sequential_decoder_academic.py) | R31-1/8-1/4 | SequentialSARDecoder | 0.9530 |    0.9073    |  0.9389   |     |  0.8002   |     0.8124     | 0.9028 |                                                          |

## Citation

```bibtex
@inproceedings{li2019show,
  title={Show, attend and read: A simple and strong baseline for irregular text recognition},
  author={Li, Hui and Wang, Peng and Shen, Chunhua and Zhang, Guyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={8610--8617},
  year={2019}
}
```
