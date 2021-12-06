# RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition

## Abstract

<!-- [ABSTRACT] -->
The attention-based encoder-decoder framework has recently achieved impressive results for scene text recognition, and many variants have emerged with improvements in recognition quality. However, it performs poorly on contextless texts (e.g., random character sequences) which is unacceptable in most of real application scenarios. In this paper, we first deeply investigate the decoding process of the decoder. We empirically find that a representative character-level sequence decoder utilizes not only context information but also positional information. Contextual information, which the existing approaches heavily rely on, causes the problem of attention drift. To suppress such side-effect, we propose a novel position enhancement branch, and dynamically fuse its outputs with those of the decoder attention module for scene text recognition. Specifically, it contains a position aware module to enable the encoder to output feature vectors encoding their own spatial positions, and an attention module to estimate glimpses using the positional clue (i.e., the current decoding time step) only. The dynamic fusion is conducted for more robust feature via an element-wise gate mechanism. Theoretically, our proposed method, dubbed \emph{RobustScanner}, decodes individual characters with dynamic ratio between context and positional clues, and utilizes more positional ones when the decoding sequences with scarce context, and thus is robust and practical. Empirically, it has achieved new state-of-the-art results on popular regular and irregular text recognition benchmarks while without much performance drop on contextless benchmarks, validating its robustness in both contextual and contextless application scenarios.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142798010-eee8795e-8cda-4a7f-a81d-ff9c94af58dc.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{yue2020robustscanner,
  title={RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition},
  author={Yue, Xiaoyu and Kuang, Zhanghui and Lin, Chenhao and Sun, Hongbin and Zhang, Wayne},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

## Dataset

### Train Dataset

|  trainset  | instance_num | repeat_num |          source          |
| :--------: | :----------: | :--------: | :----------------------: |
| icdar_2011 |     3567     |     20     |           real           |
| icdar_2013 |     848      |     20     |           real           |
| icdar2015  |     4468     |     20     |           real           |
| coco_text  |    42142     |     20     |           real           |
|   IIIT5K   |     2000     |     20     |           real           |
| SynthText  |   2400000    |     1      |          synth           |
|  SynthAdd  |   1216889    |     1      | synth, 1.6m in [[1]](#1) |
|   Syn90k   |   2400000    |     1      |          synth           |

### Test Dataset

| testset | instance_num |            type             |
| :-----: | :----------: | :-------------------------: |
| IIIT5K  |     3000     |           regular           |
|   SVT   |     647      |           regular           |
|  IC13   |     1015     |           regular           |
|  IC15   |     2077     |          irregular          |
|  SVTP   |     645      | irregular, 639 in [[1]](#1) |
|  CT80   |     288      |          irregular          |

## Results and Models

|                                     Methods                                     | GPUs |        | Regular Text |      |     |      | Irregular Text |      |                                                                                                   download                                                                                                    |
| :-----------------------------------------------------------------------------: | :--: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                 |      | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |
| [RobustScanner](configs/textrecog/robust_scanner/robustscanner_r31_academic.py) |  16  |  95.1  |     89.2     | 93.1 |     | 77.8 |      80.3      | 90.3 | [model](https://download.openmmlab.com/mmocr/textrecog/robustscanner/robustscanner_r31_academic-5f05874f.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/robustscanner/20210401_170932.log.json) |

## References

<a id="1">[1]</a> Li, Hui and Wang, Peng and Shen, Chunhua and Zhang, Guyu. Show, attend and read: A simple and strong baseline for irregular text recognition. In AAAI 2019.
