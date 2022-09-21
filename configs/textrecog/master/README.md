# MASTER

> [MASTER: Multi-aspect non-local network for scene text recognition](https://arxiv.org/abs/1910.02562)

<!-- [ALGORITHM] -->

## Abstract

Attention-based scene text recognizers have gained huge success, which leverages a more compact intermediate representation to learn 1d- or 2d- attention by a RNN-based encoder-decoder architecture. However, such methods suffer from attention-drift problem because high similarity among encoded features leads to attention confusion under the RNN-based local attention mechanism. Moreover, RNN-based methods have low efficiency due to poor parallelization. To overcome these problems, we propose the MASTER, a self-attention based scene text recognizer that (1) not only encodes the input-output attention but also learns self-attention which encodes feature-feature and target-target relationships inside the encoder and decoder and (2) learns a more powerful and robust intermediate representation to spatial distortion, and (3) owns a great training efficiency because of high training parallelization and a high-speed inference because of an efficient memory-cache mechanism. Extensive experiments on various benchmarks demonstrate the superior performance of our MASTER on both regular and irregular scene text.

<div align=center>
<img src="https://user-images.githubusercontent.com/65173622/164642001-037f81b7-37dd-4808-a6a9-09ff6f6a17ea.JPG">
</div>

## Dataset

### Train Dataset

| trainset  | instance_num | repeat_num | source |
| :-------: | :----------: | :--------: | :----: |
| SynthText |   7266686    |     1      | synth  |
| SynthAdd  |   1216889    |     1      | synth  |
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

|                               Methods                               |   Backbone    |        | Regular Text |        |     |        | Irregular Text |        |         download         |
| :-----------------------------------------------------------------: | :-----------: | :----: | :----------: | :----: | :-: | :----: | :------------: | :----: | :----------------------: |
|                                                                     |               | IIIT5K |     SVT      |  IC13  |     |  IC15  |      SVTP      |  CT80  |                          |
| [MASTER](/configs/textrecog/master/master_resnet31_12e_st_mj_sa.py) | R31-GCAModule | 0.9490 |    0.8967    | 0.9517 |     | 0.7631 |     0.8465     | 0.8854 | [model](<>) \| [log](<>) |

## Citation

```bibtex
@article{Lu2021MASTER,
  title={{MASTER}: Multi-Aspect Non-local Network for Scene Text Recognition},
  author={Ning Lu and Wenwen Yu and Xianbiao Qi and Yihao Chen and Ping Gong and Rong Xiao and Xiang Bai},
  journal={Pattern Recognition},
  year={2021}
}
```
