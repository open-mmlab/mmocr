# NRTR

## Abstract

<!-- [ABSTRACT] -->
Scene text recognition has attracted a great many researches due to its importance to various applications. Existing methods mainly adopt recurrence or convolution based networks. Though have obtained good performance, these methods still suffer from two limitations: slow training speed due to the internal recurrence of RNNs, and high complexity due to stacked convolutional layers for long-term feature extraction. This paper, for the first time, proposes a no-recurrence sequence-to-sequence text recognizer, named NRTR, that dispenses with recurrences and convolutions entirely. NRTR follows the encoder-decoder paradigm, where the encoder uses stacked self-attention to extract image features, and the decoder applies stacked self-attention to recognize texts based on encoder output. NRTR relies solely on self-attention mechanism thus could be trained with more parallelization and less complexity. Considering scene image has large variation in text and background, we further design a modality-transform block to effectively transform 2D input images to 1D sequences, combined with the encoder to extract more discriminative features. NRTR achieves state-of-the-art or highly competitive performance on both regular and irregular benchmarks, while requires only a small fraction of training time compared to the best model from the literature (at least 8 times faster).

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142797203-d9df6c35-868f-4848-8261-c286751fd342.png"/>
</div>

## Citation

Main paper

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{sheng2019nrtr,
  title={NRTR: A no-recurrence sequence-to-sequence model for scene text recognition},
  author={Sheng, Fenfen and Chen, Zhineng and Xu, Bo},
  booktitle={2019 International Conference on Document Analysis and Recognition (ICDAR)},
  pages={781--786},
  year={2019},
  organization={IEEE}
}
```

Backbone

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

|                             Methods                             |   Backbone   |        | Regular Text |      |     |      | Irregular Text |      |                                                                                               download                                                                                                |
| :-------------------------------------------------------------: | :----------: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                 |              | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |
| [NRTR](/configs/textrecog/nrtr/nrtr_r31_1by16_1by8_academic.py) | R31-1/16-1/8 |  93.9  |     90.0     | 93.5 |     | 74.5 |      78.5      | 86.5 |      [model](https://download.openmmlab.com/mmocr/textrecog/nrtr/nrtr_r31_academic_20210406-954db95e.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/nrtr/20210406_010150.log.json)      |
| [NRTR](/configs/textrecog/nrtr/nrtr_r31_1by8_1by4_academic.py)  | R31-1/8-1/4  |  94.7  |     87.5     | 93.3 |     | 75.1 |      78.9      | 87.9 | [model](https://download.openmmlab.com/mmocr/textrecog/nrtr/nrtr_r31_1by8_1by4_academic_20210406-ce16e7cc.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/nrtr/20210406_160845.log.json) |

**Notes:**

-   `R31-1/16-1/8` means the height of feature from backbone is 1/16 of input image, where 1/8 for width.
-   `R31-1/8-1/4` means the height of feature from backbone is 1/8 of input image, where 1/4 for width.
