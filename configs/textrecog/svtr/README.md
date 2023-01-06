# SVTR

> [SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)

<!-- [ALGORITHM] -->

## Abstract

Dominant scene text recognition models commonly contain two building blocks, a visual model for feature extraction and a sequence model for text transcription. This hybrid architecture, although accurate, is complex and less efficient. In this study, we propose a Single Visual model for Scene Text recognition within the patch-wise image tokenization framework, which dispenses with the sequential modeling entirely. The method, termed SVTR, firstly decomposes an image text into small patches named character components. Afterward, hierarchical stages are recurrently carried out by component-level mixing, merging and/or combining. Global and local mixing blocks are devised to perceive the inter-character and intra-character patterns, leading to a multi-grained character component perception. Thus, characters are recognized by a simple linear prediction. Experimental results on both English and Chinese scene text recognition tasks demonstrate the effectiveness of SVTR. SVTR-L (Large) achieves highly competitive accuracy in English and outperforms existing methods by a large margin in Chinese, while running faster. In addition, SVTR-T (Tiny) is an effective and much smaller model, which shows appealing speed at inference.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/210541576-025df5d5-f4d2-4037-82e0-246cf8cd3c25.png"/>
</div>

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

|                            Methods                            |        | Regular Text |           |     |           | Irregular Text |        |                                     download                                     |
| :-----------------------------------------------------------: | :----: | :----------: | :-------: | :-: | :-------: | :------------: | :----: | :------------------------------------------------------------------------------: |
|                                                               | IIIT5K |     SVT      | IC13-1015 |     | IC15-2077 |      SVTP      |  CT80  |                                                                                  |
|  [SVTR-tiny](/configs/textrecog/svtr/svtr-tiny_20e_st_mj.py)  |   -    |      -       |     -     |     |     -     |       -        |   -    |                             [model](<>) \| [log](<>)                             |
| [SVTR-small](/configs/textrecog/svtr/svtr-small_20e_st_mj.py) | 0.8553 |    0.9026    |  0.9448   |     |  0.7496   |     0.8496     | 0.8854 | [model](https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-small_20e_st_mj/svtr-small_20e_st_mj-35d800d6.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-small_20e_st_mj/20230105_184454.log) |
|  [SVTR-base](/configs/textrecog/svtr/svtr-base_20e_st_mj.py)  | 0.8570 |    0.9181    |  0.9438   |     |  0.7448   |     0.8388     | 0.9028 | [model](https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-base_20e_st_mj/svtr-base_20e_st_mj-ea500101.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-base_20e_st_mj/20221227_175415.log) |
| [SVTR-large](/configs/textrecog/svtr/svtr-large_20e_st_mj.py) |   -    |      -       |     -     |     |     -     |       -        |   -    |                             [model](<>) \| [log](<>)                             |

```{note}
The implementation and configuration follow the original code and paper, but there is still a gap between the reproduced results and the official ones. We appreciate any suggestions to improve its performance.
```

## Citation

```bibtex
@inproceedings{ijcai2022p124,
  title     = {SVTR: Scene Text Recognition with a Single Visual Model},
  author    = {Du, Yongkun and Chen, Zhineng and Jia, Caiyan and Yin, Xiaoting and Zheng, Tianlun and Li, Chenxia and Du, Yuning and Jiang, Yu-Gang},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {884--890},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/124},
  url       = {https://doi.org/10.24963/ijcai.2022/124},
}

```
