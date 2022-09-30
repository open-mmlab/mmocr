# ABINet

> [Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition](https://arxiv.org/abs/2103.06495)

<!-- [ALGORITHM] -->

## Abstract

Linguistic knowledge is of great benefit to scene text recognition. However, how to effectively model linguistic rules in end-to-end deep networks remains a research challenge. In this paper, we argue that the limited capacity of language models comes from: 1) implicitly language modeling; 2) unidirectional feature representation; and 3) language model with noise input. Correspondingly, we propose an autonomous, bidirectional and iterative ABINet for scene text recognition. Firstly, the autonomous suggests to block gradient flow between vision and language models to enforce explicitly language modeling. Secondly, a novel bidirectional cloze network (BCN) as the language model is proposed based on bidirectional feature representation. Thirdly, we propose an execution manner of iterative correction for language model which can effectively alleviate the impact of noise input. Additionally, based on the ensemble of iterative predictions, we propose a self-training method which can learn from unlabeled images effectively. Extensive experiments indicate that ABINet has superiority on low-quality images and achieves state-of-the-art results on several mainstream benchmarks. Besides, the ABINet trained with ensemble self-training shows promising improvement in realizing human-level recognition.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/145804331-9ae955dc-0d3b-41eb-a6b2-dc7c9f7c1bef.png"/>
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

|                                methods                                | pretrained |        | Regular Text |        |        | Irregular Text |        | download                                                                |
| :-------------------------------------------------------------------: | :--------: | :----: | :----------: | :----: | :----: | :------------: | :----: | :---------------------------------------------------------------------- |
|                                                                       |            | IIIT5K |     SVT      |  IC13  |  IC15  |      SVTP      |  CT80  |                                                                         |
| [ABINet-Vision](/configs/textrecog/abinet/abinet-vision_20e_st-an_mj.py) |     -      | 0.9523 |    0.9057    | 0.9369 | 0.7886 |     0.8403     | 0.8437 | [model](https://download.openmmlab.com/mmocr/textrecog/abinet/abinet-vision_20e_st-an_mj/abinet-vision_20e_st-an_mj_20220915_152445-85cfb03d.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/abinet/abinet-vision_20e_st-an_mj/20220915_152445.log) |

```{note}
1. ABINet allows its encoder to run and be trained without decoder and fuser. Its encoder is designed to recognize texts as a stand-alone model and therefore can work as an independent text recognizer. We release it as ABINet-Vision.
2. Facts about the pretrained model: MMOCR does not have a systematic pipeline to pretrain the language model (LM) yet, thus the weights of LM are converted from [the official pretrained model](https://github.com/FangShancheng/ABINet). The weights of ABINet-Vision are directly used as the vision model of ABINet.
```

## Citation

```bibtex
@article{fang2021read,
  title={Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition},
  author={Fang, Shancheng and Xie, Hongtao and Wang, Yuxin and Mao, Zhendong and Zhang, Yongdong},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
