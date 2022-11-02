# oCLIP

> [Language Matters: A Weakly Supervised Vision-Language Pre-training Approach for Scene Text Detection and Spotting](https://arxiv.org/abs/2203.03911)

<!-- [ALGORITHM] -->

## Abstract

Recently, Vision-Language Pre-training (VLP) techniques have greatly benefited various vision-language tasks by jointly learning visual and textual representations, which intuitively helps in Optical Character Recognition (OCR) tasks due to the rich visual and textual information in scene text images. However, these methods cannot well cope with OCR tasks because of the difficulty in both instance-level text encoding and image-text pair acquisition (i.e. images and captured texts in them). This paper presents a weakly supervised pre-training method, oCLIP, which can acquire effective scene text representations by jointly learning and aligning visual and textual information. Our network consists of an image encoder and a character-aware text encoder that extract visual and textual features, respectively, as well as a visual-textual decoder that models the interaction among textual and visual features for learning effective scene text representations. With the learning of textual features, the pre-trained model can attend texts in images well with character awareness. Besides, these designs enable the learning from weakly annotated texts (i.e. partial texts in images without text bounding boxes) which mitigates the data annotation constraint greatly. Experiments over the weakly annotated images in ICDAR2019-LSVT show that our pre-trained model improves F-score by +2.5% and +4.8% while transferring its weights to other text detection and spotting networks, respectively. In addition, the proposed method outperforms existing pre-training techniques consistently across multiple public datasets (e.g., +3.2% and +1.3% for Total-Text and CTW1500).

<div align=center>
<img src="https://user-images.githubusercontent.com/24622904/199475057-aa688422-518d-4d7a-86fc-1be0cc1b5dc6.png"/>
</div>

## Support Model

|           | DBNet | DBNet++ | FCENet | TextSnake | PSENet | DRRG | Mask R-CNN |
| :-------: | :---: | :-----: | :----: | :-------: | :----: | :--: | :--------: |
| ICDAR2015 |   ✓   |    ✓    |   ✓    |           |   ✓    |      |     ✓      |
|  CTW1500  |       |         |        |     ✓     |   ✓    |  ✓   |     ✓      |

## Citation

```bibtex
@article{xue2022language,
  title={Language Matters: A Weakly Supervised Pre-training Approach for Scene Text Detection and Spotting},
  author={Xue, Chuhui and Zhang, Wenqing and Hao, Yu and Lu, Shijian and Torr, Philip and Bai, Song},
  journal={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
