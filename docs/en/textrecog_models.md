# Text Recognition Models

## ABINet

[Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition](https://arxiv.org/abs/2103.06495)

<!-- [ALGORITHM] -->

### Abstract

Linguistic knowledge is of great benefit to scene text recognition. However, how to effectively model linguistic rules in end-to-end deep networks remains a research challenge. In this paper, we argue that the limited capacity of language models comes from: 1) implicitly language modeling; 2) unidirectional feature representation; and 3) language model with noise input. Correspondingly, we propose an autonomous, bidirectional and iterative ABINet for scene text recognition. Firstly, the autonomous suggests to block gradient flow between vision and language models to enforce explicitly language modeling. Secondly, a novel bidirectional cloze network (BCN) as the language model is proposed based on bidirectional feature representation. Thirdly, we propose an execution manner of iterative correction for language model which can effectively alleviate the impact of noise input. Additionally, based on the ensemble of iterative predictions, we propose a self-training method which can learn from unlabeled images effectively. Extensive experiments indicate that ABINet has superiority on low-quality images and achieves state-of-the-art results on several mainstream benchmarks. Besides, the ABINet trained with ensemble self-training shows promising improvement in realizing human-level recognition.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/145804331-9ae955dc-0d3b-41eb-a6b2-dc7c9f7c1bef.png"/>
</div>

### Dataset

#### Train Dataset

| trainset  | instance_num | repeat_num |     note     |
| :-------: | :----------: | :--------: | :----------: |
|  Syn90k   |   8919273    |     1      |    synth     |
| SynthText |   7239272    |     1      | alphanumeric |

#### Test Dataset

| testset | instance_num |   note    |
| :-----: | :----------: | :-------: |
| IIIT5K  |     3000     |  regular  |
|   SVT   |     647      |  regular  |
|  IC13   |     1015     |  regular  |
|  IC15   |     2077     | irregular |
|  SVTP   |     645      | irregular |
|  CT80   |     288      | irregular |

### Results and models

|                      methods                       |                       pretrained                       |        | Regular Text |      |      | Irregular Text |      | download                                             |
| :------------------------------------------------: | :----------------------------------------------------: | :----: | :----------: | :--: | :--: | :------------: | :--: | :--------------------------------------------------- |
|                                                    |                                                        | IIIT5K |     SVT      | IC13 | IC15 |      SVTP      | CT80 |                                                      |
| [ABINet-Vision](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/abinet/abinet-vision_6e_st-an_mj.py) |                           -                            |  94.7  |     91.7     | 93.6 | 83.0 |      85.1      | 86.5 | [model](https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_vision_only_academic-e6b9ea89.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/abinet/20211201_195512.log) |
| [ABINet](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/abinet/abinet_6e_st-an_mj.py) | [Pretrained](https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_pretrain-1bed979b.pth) |  95.7  |     94.6     | 95.7 | 85.1 |      90.4      | 90.3 | [model](https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_academic-f718abf6.pth) \| [log1](https://download.openmmlab.com/mmocr/textrecog/abinet/20211210_095832.log) \| [log2](https://download.openmmlab.com/mmocr/textrecog/abinet/20211213_131724.log) |

```{note}
1. ABINet allows its encoder to run and be trained without decoder and fuser. Its encoder is designed to recognize texts as a stand-alone model and therefore can work as an independent text recognizer. We release it as ABINet-Vision.
2. Facts about the pretrained model: MMOCR does not have a systematic pipeline to pretrain the language model (LM) yet, thus the weights of LM are converted from [the official pretrained model](https://github.com/FangShancheng/ABINet). The weights of ABINet-Vision are directly used as the vision model of ABINet.
3. Due to some technical issues, the training process of ABINet was interrupted at the 13th epoch and we resumed it later. Both logs are released for full reference.
4. The model architecture in the logs looks slightly different from the final released version, since it was refactored afterward. However, both architectures are essentially equivalent.
```

### Citation

```bibtex
@article{fang2021read,
  title={Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition},
  author={Fang, Shancheng and Xie, Hongtao and Wang, Yuxin and Mao, Zhendong and Zhang, Yongdong},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## CRNN

[An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/abs/1507.05717)

<!-- [ALGORITHM] -->

### Abstract

Image-based sequence recognition has been a long-standing research topic in computer vision. In this paper, we investigate the problem of scene text recognition, which is among the most important and challenging tasks in image-based sequence recognition. A novel neural network architecture, which integrates feature extraction, sequence modeling and transcription into a unified framework, is proposed. Compared with previous systems for scene text recognition, the proposed architecture possesses four distinctive properties: (1) It is end-to-end trainable, in contrast to most of the existing algorithms whose components are separately trained and tuned. (2) It naturally handles sequences in arbitrary lengths, involving no character segmentation or horizontal scale normalization. (3) It is not confined to any predefined lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene text recognition tasks. (4) It generates an effective yet much smaller model, which is more practical for real-world application scenarios. The experiments on standard benchmarks, including the IIIT-5K, Street View Text and ICDAR datasets, demonstrate the superiority of the proposed algorithm over the prior arts. Moreover, the proposed algorithm performs well in the task of image-based music score recognition, which evidently verifies the generality of it.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142797788-6b1cd78d-1dd6-4e02-be32-3dbd257c4992.png"/>
</div>

### Dataset

#### Train Dataset

| trainset | instance_num | repeat_num | note  |
| :------: | :----------: | :--------: | :---: |
|  Syn90k  |   8919273    |     1      | synth |

#### Test Dataset

| testset | instance_num |   note    |
| :-----: | :----------: | :-------: |
| IIIT5K  |     3000     |  regular  |
|   SVT   |     647      |  regular  |
|  IC13   |     1015     |  regular  |
|  IC15   |     2077     | irregular |
|  SVTP   |     645      | irregular |
|  CT80   |     288      | irregular |

### Results and models

|                                   methods                                    |        | Regular Text |      |     |      | Irregular Text |      |                                   download                                    |
| :--------------------------------------------------------------------------: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :---------------------------------------------------------------------------: |
|                                   methods                                    | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |                                                                               |
| [CRNN](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/crnn/crnn_mini-vgg_5e_mj.py) |  80.5  |     81.5     | 86.5 |     | 54.1 |      59.1      | 55.6 | [model](https://download.openmmlab.com/mmocr/textrecog/crnn/crnn_academic-a723a1c5.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/crnn/20210326_111035.log.json) |

### Citation

```bibtex
@article{shi2016end,
  title={An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition},
  author={Shi, Baoguang and Bai, Xiang and Yao, Cong},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2016}
}
```

## MASTER

[MASTER: Multi-aspect non-local network for scene text recognition](https://arxiv.org/abs/1910.02562)

<!-- [ALGORITHM] -->

### Abstract

Attention-based scene text recognizers have gained huge success, which leverages a more compact intermediate representation to learn 1d- or 2d- attention by a RNN-based encoder-decoder architecture. However, such methods suffer from attention-drift problem because high similarity among encoded features leads to attention confusion under the RNN-based local attention mechanism. Moreover, RNN-based methods have low efficiency due to poor parallelization. To overcome these problems, we propose the MASTER, a self-attention based scene text recognizer that (1) not only encodes the input-output attention but also learns self-attention which encodes feature-feature and target-target relationships inside the encoder and decoder and (2) learns a more powerful and robust intermediate representation to spatial distortion, and (3) owns a great training efficiency because of high training parallelization and a high-speed inference because of an efficient memory-cache mechanism. Extensive experiments on various benchmarks demonstrate the superior performance of our MASTER on both regular and irregular scene text.

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/164642001-037f81b7-37dd-4808-a6a9-09ff6f6a17ea.JPG">
</div>

### Dataset

#### Train Dataset

| trainset  | instance_num | repeat_num | source |
| :-------: | :----------: | :--------: | :----: |
| SynthText |   7266686    |     1      | synth  |
| SynthAdd  |   1216889    |     1      | synth  |
|  Syn90k   |   8919273    |     1      | synth  |

#### Test Dataset

| testset | instance_num |   type    |
| :-----: | :----------: | :-------: |
| IIIT5K  |     3000     |  regular  |
|   SVT   |     647      |  regular  |
|  IC13   |     1015     |  regular  |
|  IC15   |     2077     | irregular |
|  SVTP   |     645      | irregular |
|  CT80   |     288      | irregular |

### Results and Models

|                               Methods                                |   Backbone    |        | Regular Text |       |     |       | Irregular Text |       |                               download                                |
| :------------------------------------------------------------------: | :-----------: | :----: | :----------: | :---: | :-: | :---: | :------------: | :---: | :-------------------------------------------------------------------: |
|                                                                      |               | IIIT5K |     SVT      | IC13  |     | IC15  |      SVTP      | CT80  |                                                                       |
| [MASTER](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/master/master_resnet31_12e_st_mj_sa.py) | R31-GCAModule | 94.63  |    90.42     | 94.98 |     | 75.54 |     82.79      | 88.54 | [model](https://download.openmmlab.com/mmocr/textrecog/master/master_r31_12e_ST_MJ_SA-787edd36.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/master/master_r31_12e_ST_MJ_SA-787edd36.log.json) |

### Citation

```bibtex
@article{Lu2021MASTER,
  title={{MASTER}: Multi-Aspect Non-local Network for Scene Text Recognition},
  author={Ning Lu and Wenwen Yu and Xianbiao Qi and Yihao Chen and Ping Gong and Rong Xiao and Xiang Bai},
  journal={Pattern Recognition},
  year={2021}
}
```

## NRTR

[NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition](https://arxiv.org/abs/1806.00926)

<!-- [ALGORITHM] -->

### Abstract

Scene text recognition has attracted a great many researches due to its importance to various applications. Existing methods mainly adopt recurrence or convolution based networks. Though have obtained good performance, these methods still suffer from two limitations: slow training speed due to the internal recurrence of RNNs, and high complexity due to stacked convolutional layers for long-term feature extraction. This paper, for the first time, proposes a no-recurrence sequence-to-sequence text recognizer, named NRTR, that dispenses with recurrences and convolutions entirely. NRTR follows the encoder-decoder paradigm, where the encoder uses stacked self-attention to extract image features, and the decoder applies stacked self-attention to recognize texts based on encoder output. NRTR relies solely on self-attention mechanism thus could be trained with more parallelization and less complexity. Considering scene image has large variation in text and background, we further design a modality-transform block to effectively transform 2D input images to 1D sequences, combined with the encoder to extract more discriminative features. NRTR achieves state-of-the-art or highly competitive performance on both regular and irregular benchmarks, while requires only a small fraction of training time compared to the best model from the literature (at least 8 times faster).

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142797203-d9df6c35-868f-4848-8261-c286751fd342.png"/>
</div>

### Dataset

#### Train Dataset

| trainset  | instance_num | repeat_num | source |
| :-------: | :----------: | :--------: | :----: |
| SynthText |   7266686    |     1      | synth  |
|  Syn90k   |   8919273    |     1      | synth  |

#### Test Dataset

| testset | instance_num |   type    |
| :-----: | :----------: | :-------: |
| IIIT5K  |     3000     |  regular  |
|   SVT   |     647      |  regular  |
|  IC13   |     1015     |  regular  |
|  IC15   |     2077     | irregular |
|  SVTP   |     645      | irregular |
|  CT80   |     288      | irregular |

### Results and Models

|                               Methods                                |   Backbone   |        | Regular Text |       |     |       | Irregular Text |       |                                download                                |
| :------------------------------------------------------------------: | :----------: | :----: | :----------: | :---: | :-: | :---: | :------------: | :---: | :--------------------------------------------------------------------: |
|                                                                      |              | IIIT5K |     SVT      | IC13  |     | IC15  |      SVTP      | CT80  |                                                                        |
| [NRTR](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/nrtr/nrtr_resnet31-1by16-1by8_6e_st_mj.py) | R31-1/16-1/8 |  94.8  |    89.03     | 93.79 |     | 74.19 |     80.31      | 87.15 | [model](https://download.openmmlab.com/mmocr/textrecog/nrtr/nrtr_r31_1by16_1by8_academic_20211124-f60cebf4.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/nrtr/20211124_002420.log.json) |
| [NRTR](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/nrtr/nrtr_resnet31-1by8-1by4_6e_st_mj.py) | R31-1/8-1/4  |  95.5  |    90.01     | 94.38 |     | 74.05 |     79.53      | 87.15 | [model](https://download.openmmlab.com/mmocr/textrecog/nrtr/nrtr_r31_1by8_1by4_academic_20211123-e1fdb322.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/nrtr/20211123_232151.log.json) |

```{note}

- For backbone `R31-1/16-1/8`:
  - The output consists of 92 classes, including 26 lowercase letters, 26 uppercase letters, 28 symbols, 10 digital numbers, 1 unknown token and 1 end-of-sequence token.
  - The encoder-block number is 6.
  - `1/16-1/8` means the height of feature from backbone is 1/16 of input image, where 1/8 for width.
- For backbone `R31-1/8-1/4`:
  - The output consists of 92 classes, including 26 lowercase letters, 26 uppercase letters, 28 symbols, 10 digital numbers, 1 unknown token and 1 end-of-sequence token.
  - The encoder-block number is 6.
  - `1/8-1/4` means the height of feature from backbone is 1/8 of input image, where 1/4 for width.
```

### Citation

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

## RobustScanner

[RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition](https://arxiv.org/abs/2007.07542)

<!-- [ALGORITHM] -->

### Abstract

The attention-based encoder-decoder framework has recently achieved impressive results for scene text recognition, and many variants have emerged with improvements in recognition quality. However, it performs poorly on contextless texts (e.g., random character sequences) which is unacceptable in most of real application scenarios. In this paper, we first deeply investigate the decoding process of the decoder. We empirically find that a representative character-level sequence decoder utilizes not only context information but also positional information. Contextual information, which the existing approaches heavily rely on, causes the problem of attention drift. To suppress such side-effect, we propose a novel position enhancement branch, and dynamically fuse its outputs with those of the decoder attention module for scene text recognition. Specifically, it contains a position aware module to enable the encoder to output feature vectors encoding their own spatial positions, and an attention module to estimate glimpses using the positional clue (i.e., the current decoding time step) only. The dynamic fusion is conducted for more robust feature via an element-wise gate mechanism. Theoretically, our proposed method, dubbed \\emph{RobustScanner}, decodes individual characters with dynamic ratio between context and positional clues, and utilizes more positional ones when the decoding sequences with scarce context, and thus is robust and practical. Empirically, it has achieved new state-of-the-art results on popular regular and irregular text recognition benchmarks while without much performance drop on contextless benchmarks, validating its robustness in both contextual and contextless application scenarios.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142798010-eee8795e-8cda-4a7f-a81d-ff9c94af58dc.png"/>
</div>

### Dataset

#### Train Dataset

|  trainset  | instance_num | repeat_num |           source            |
| :--------: | :----------: | :--------: | :-------------------------: |
| icdar_2011 |     3567     |     20     |            real             |
| icdar_2013 |     848      |     20     |            real             |
| icdar2015  |     4468     |     20     |            real             |
| coco_text  |    42142     |     20     |            real             |
|   IIIT5K   |     2000     |     20     |            real             |
| SynthText  |   2400000    |     1      |            synth            |
|  SynthAdd  |   1216889    |     1      | synth, 1.6m in [\[1\]](##1) |
|   Syn90k   |   2400000    |     1      |            synth            |

#### Test Dataset

| testset | instance_num |              type              |
| :-----: | :----------: | :----------------------------: |
| IIIT5K  |     3000     |            regular             |
|   SVT   |     647      |            regular             |
|  IC13   |     1015     |            regular             |
|  IC15   |     2077     |           irregular            |
|  SVTP   |     645      | irregular, 639 in [\[1\]](##1) |
|  CT80   |     288      |           irregular            |

### Results and Models

|                                  Methods                                   | GPUs |        | Regular Text |      |     |      | Irregular Text |      |                                  download                                   |
| :------------------------------------------------------------------------: | :--: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :-------------------------------------------------------------------------: |
|                                                                            |      | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |                                                                             |
| [RobustScanner](configs/textrecog/robust_scanner/robustscanner_resnet31_5e_st-sub_mj-sub_sa_real.py) |  16  |  95.1  |     89.2     | 93.1 |     | 77.8 |      80.3      | 90.3 | [model](https://download.openmmlab.com/mmocr/textrecog/robustscanner/robustscanner_r31_academic-5f05874f.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/robustscanner/20210401_170932.log.json) |

### References

<a id="1">\[1\]</a> Li, Hui and Wang, Peng and Shen, Chunhua and Zhang, Guyu. Show, attend and read: A simple and strong baseline for irregular text recognition. In AAAI 2019.

### Citation

```bibtex
@inproceedings{yue2020robustscanner,
  title={RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition},
  author={Yue, Xiaoyu and Kuang, Zhanghui and Lin, Chenhao and Sun, Hongbin and Zhang, Wayne},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

## SAR

[Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/abs/1811.00751)

<!-- [ALGORITHM] -->

### Abstract

Recognizing irregular text in natural scene images is challenging due to the large variance in text appearance, such as curvature, orientation and distortion. Most existing approaches rely heavily on sophisticated model designs and/or extra fine-grained annotations, which, to some extent, increase the difficulty in algorithm implementation and data collection. In this work, we propose an easy-to-implement strong baseline for irregular scene text recognition, using off-the-shelf neural network components and only word-level annotations. It is composed of a 31-layer ResNet, an LSTM-based encoder-decoder framework and a 2-dimensional attention module. Despite its simplicity, the proposed method is robust and achieves state-of-the-art performance on both regular and irregular scene text recognition benchmarks.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142798157-ac68907f-5a8a-473f-a29f-f0532b7fdba0.png"/>
</div>

### Dataset

#### Train Dataset

|  trainset  | instance_num | repeat_num |           source            |
| :--------: | :----------: | :--------: | :-------------------------: |
| icdar_2011 |     3567     |     20     |            real             |
| icdar_2013 |     848      |     20     |            real             |
| icdar2015  |     4468     |     20     |            real             |
| coco_text  |    42142     |     20     |            real             |
|   IIIT5K   |     2000     |     20     |            real             |
| SynthText  |   2400000    |     1      |            synth            |
|  SynthAdd  |   1216889    |     1      | synth, 1.6m in [\[1\]](##1) |
|   Syn90k   |   2400000    |     1      |            synth            |

#### Test Dataset

| testset | instance_num |              type              |
| :-----: | :----------: | :----------------------------: |
| IIIT5K  |     3000     |            regular             |
|   SVT   |     647      |            regular             |
|  IC13   |     1015     |            regular             |
|  IC15   |     2077     |           irregular            |
|  SVTP   |     645      | irregular, 639 in [\[1\]](##1) |
|  CT80   |     288      |           irregular            |

### Results and Models

|                           Methods                            |  Backbone   |       Decoder        |        | Regular Text |      |     |      | Irregular Text |      |                            download                            |
| :----------------------------------------------------------: | :---------: | :------------------: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :------------------------------------------------------------: |
|                                                              |             |                      | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |                                                                |
| [SAR](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py) | R31-1/8-1/4 |  ParallelSARDecoder  |  95.0  |     89.6     | 93.7 |     | 79.0 |      82.2      | 88.9 | [model](https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/sar/20210327_154129.log.json) |
| [SAR](configs/textrecog/sar/sar_r31_sequential_decoder_academic.py) | R31-1/8-1/4 | SequentialSARDecoder |  95.2  |     88.7     | 92.4 |     | 78.2 |      81.9      | 89.6 | [model](https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_sequential_decoder_academic-d06c9a8e.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/sar/20210330_105728.log.json) |

### Chinese Dataset

### Results and Models

|                                       Methods                                       |  Backbone   |      Decoder       |     |                                       download                                        |
| :---------------------------------------------------------------------------------: | :---------: | :----------------: | :-: | :-----------------------------------------------------------------------------------: |
| [SAR](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/sar/sar_r31_parallel_decoder_chinese.py) | R31-1/8-1/4 | ParallelSARDecoder |     | [model](https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/sar/20210506_225557.log.json) \| [dict](https://download.openmmlab.com/mmocr/textrecog/sar/dict_printed_chinese_english_digits.txt) |

```{note}

-   `R31-1/8-1/4` means the height of feature from backbone is 1/8 of input image, where 1/4 for width.
-   We did not use beam search during decoding.
-   We implemented two kinds of decoder. Namely, `ParallelSARDecoder` and `SequentialSARDecoder`.
    -   `ParallelSARDecoder`: Parallel decoding during training with `LSTM` layer. It would be faster.
    -   `SequentialSARDecoder`: Sequential Decoding during training with `LSTMCell`. It would be easier to understand.
-   For train dataset.
    -   We did not construct distinct data groups (20 groups in [[1]](##1)) to train the model group-by-group since it would render model training too complicated.
    -   Instead, we randomly selected `2.4m` patches from `Syn90k`, `2.4m` from `SynthText` and `1.2m` from `SynthAdd`, and grouped all data together. See [config](https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_academic.py) for details.
-   We used 48 GPUs with `total_batch_size = 64 * 48` in the experiment above to speedup training, while keeping the `initial lr = 1e-3` unchanged.
```

### Citation

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

## SATRN

[On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://arxiv.org/abs/1910.04396)

<!-- [ALGORITHM] -->

### Abstract

Scene text recognition (STR) is the task of recognizing character sequences in natural scenes. While there have been great advances in STR methods, current methods still fail to recognize texts in arbitrary shapes, such as heavily curved or rotated texts, which are abundant in daily life (e.g. restaurant signs, product labels, company logos, etc). This paper introduces a novel architecture to recognizing texts of arbitrary shapes, named Self-Attention Text Recognition Network (SATRN), which is inspired by the Transformer. SATRN utilizes the self-attention mechanism to describe two-dimensional (2D) spatial dependencies of characters in a scene text image. Exploiting the full-graph propagation of self-attention, SATRN can recognize texts with arbitrary arrangements and large inter-character spacing. As a result, SATRN outperforms existing STR models by a large margin of 5.7 pp on average in "irregular text" benchmarks. We provide empirical analyses that illustrate the inner mechanisms and the extent to which the model is applicable (e.g. rotated and multi-line text). We will open-source the code.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142798828-cc4ded5d-3fb8-478c-9f3e-74edbcf41982.png"/>
</div>

### Dataset

#### Train Dataset

| trainset  | instance_num | repeat_num | source |
| :-------: | :----------: | :--------: | :----: |
| SynthText |   7266686    |     1      | synth  |
|  Syn90k   |   8919273    |     1      | synth  |

#### Test Dataset

| testset | instance_num |   type    |
| :-----: | :----------: | :-------: |
| IIIT5K  |     3000     |  regular  |
|   SVT   |     647      |  regular  |
|  IC13   |     1015     |  regular  |
|  IC15   |     2077     | irregular |
|  SVTP   |     645      | irregular |
|  CT80   |     288      | irregular |

### Results and Models

|                                   Methods                                    |        | Regular Text |      |     |      | Irregular Text |      |                                   download                                    |
| :--------------------------------------------------------------------------: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :---------------------------------------------------------------------------: |
|                                                                              | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |                                                                               |
| [Satrn](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/satrn/satrn_shallow_5e_st_mj.py) |  95.1  |     92.0     | 95.8 |     | 81.4 |      87.6      | 90.6 | [model](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_academic_20211009-cb8b1580.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/satrn/20210809_093244.log.json) |
| [Satrn_small](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/satrn/satrn_shallow-small_5e_st_mj.py) |  94.7  |     91.3     | 95.4 |     | 81.9 |      85.9      | 86.5 | [model](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_small_20211009-2cf13355.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/satrn/20210811_053047.log.json) |

### Citation

```bibtex
@article{junyeop2019recognizing,
  title={On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention},
  author={Junyeop Lee, Sungrae Park, Jeonghun Baek, Seong Joon Oh, Seonghyeon Kim, Hwalsuk Lee},
  year={2019}
}
```

## CRNN-STN

<!-- [ALGORITHM] -->

### Abstract

Image-based sequence recognition has been a long-standing research topic in computer vision. In this paper, we investigate the problem of scene text recognition, which is among the most important and challenging tasks in image-based sequence recognition. A novel neural network architecture, which integrates feature extraction, sequence modeling and transcription into a unified framework, is proposed. Compared with previous systems for scene text recognition, the proposed architecture possesses four distinctive properties: (1) It is end-to-end trainable, in contrast to most of the existing algorithms whose components are separately trained and tuned. (2) It naturally handles sequences in arbitrary lengths, involving no character segmentation or horizontal scale normalization. (3) It is not confined to any predefined lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene text recognition tasks. (4) It generates an effective yet much smaller model, which is more practical for real-world application scenarios. The experiments on standard benchmarks, including the IIIT-5K, Street View Text and ICDAR datasets, demonstrate the superiority of the proposed algorithm over the prior arts. Moreover, the proposed algorithm performs well in the task of image-based music score recognition, which evidently verifies the generality of it.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142797788-6b1cd78d-1dd6-4e02-be32-3dbd257c4992.png"/>
</div>

```{note}
We use STN from this paper as the preprocessor and CRNN as the recognition network.
```

### Dataset

#### Train Dataset

| trainset | instance_num | repeat_num | note  |
| :------: | :----------: | :--------: | :---: |
|  Syn90k  |   8919273    |     1      | synth |

#### Test Dataset

| testset | instance_num |   note    |
| :-----: | :----------: | :-------: |
| IIIT5K  |     3000     |  regular  |
|   SVT   |     647      |  regular  |
|  IC13   |     1015     |  regular  |
|  IC15   |     2077     | irregular |
|  SVTP   |     645      | irregular |
|  CT80   |     288      | irregular |

### Results and models

|                                   methods                                    |        | Regular Text |      |     |      | Irregular Text |      |                                   download                                    |
| :--------------------------------------------------------------------------: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :---------------------------------------------------------------------------: |
|                                                                              | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |                                                                               |
| [CRNN-STN](https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/tps/crnn_tps_academic_dataset.py) |  80.8  |     81.3     | 85.0 |     | 59.6 |      68.1      | 53.8 | [model](https://download.openmmlab.com/mmocr/textrecog/tps/crnn_tps_academic_dataset_20210510-d221a905.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/tps/20210510_204353.log.json) |

### Citation

```bibtex
@article{shi2016robust,
  title={Robust Scene Text Recognition with Automatic Rectification},
  author={Shi, Baoguang and Wang, Xinggang and Lyu, Pengyuan and Yao,
  Cong and Bai, Xiang},
  year={2016}
}
```
