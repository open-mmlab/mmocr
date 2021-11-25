# CRNN with TPS based STN

## Abstract

<!-- [ABSTRACT] -->
Image-based sequence recognition has been a long-standing research topic in computer vision. In this paper, we investigate the problem of scene text recognition, which is among the most important and challenging tasks in image-based sequence recognition. A novel neural network architecture, which integrates feature extraction, sequence modeling and transcription into a unified framework, is proposed. Compared with previous systems for scene text recognition, the proposed architecture possesses four distinctive properties: (1) It is end-to-end trainable, in contrast to most of the existing algorithms whose components are separately trained and tuned. (2) It naturally handles sequences in arbitrary lengths, involving no character segmentation or horizontal scale normalization. (3) It is not confined to any predefined lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene text recognition tasks. (4) It generates an effective yet much smaller model, which is more practical for real-world application scenarios. The experiments on standard benchmarks, including the IIIT-5K, Street View Text and ICDAR datasets, demonstrate the superiority of the proposed algorithm over the prior arts. Moreover, the proposed algorithm performs well in the task of image-based music score recognition, which evidently verifies the generality of it.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142797788-6b1cd78d-1dd6-4e02-be32-3dbd257c4992.png"/>
</div>

## Citation

Main paper

```bibtex
@article{shi2016end,
  title={An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition},
  author={Shi, Baoguang and Bai, Xiang and Yao, Cong},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2016}
}
```

Preprocessor

<!-- [PREPROCESSOR] -->

```bibtex
@article{shi2016robust,
  title={Robust Scene Text Recognition with Automatic Rectification},
  author={Shi, Baoguang and Wang, Xinggang and Lyu, Pengyuan and Yao,
  Cong and Bai, Xiang},
  year={2016}
}
```

## Results and Models

### Train Dataset

| trainset | instance_num | repeat_num | note  |
| :------: | :----------: | :--------: | :---: |
|  Syn90k  |   8919273    |     1      | synth |

### Test Dataset

| testset | instance_num |  note   |
| :-----: | :----------: | :-----: |
| IIIT5K  |     3000     | regular |
|   SVT   |     647      | regular |
|  IC13   |     1015     | regular |
|  IC15   |     2077     |irregular|
|  SVTP   |     645      |irregular|
|  CT80   |     288      |irregular|

## Results and models

|                         methods                          |        | Regular Text |      |     |      | Irregular Text |      |                                                                                    download                                                                                    |
| :------------------------------------------------------: | :----: | :----------: | :--: | :-: | :--: | :------------: | :--: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                   | IIIT5K |     SVT      | IC13 |     | IC15 |      SVTP      | CT80 |
| [CRNN-STN](/configs/textrecog/tps/crnn_tps_academic_dataset.py) |  80.8  |     81.3     | 85.0 |     |  59.6   |       68.1        |  53.8   | [model](https://download.openmmlab.com/mmocr/textrecog/tps/crnn_tps_academic_dataset_20210510-d221a905.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/tps/20210510_204353.log.json) |
