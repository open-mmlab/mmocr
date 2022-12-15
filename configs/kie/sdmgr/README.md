# SDMGR

> [Spatial Dual-Modality Graph Reasoning for Key Information Extraction](https://arxiv.org/abs/2103.14470)

<!-- [ALGORITHM] -->

## Abstract

Key information extraction from document images is of paramount importance in office automation. Conventional template matching based approaches fail to generalize well to document images of unseen templates, and are not robust against text recognition errors. In this paper, we propose an end-to-end Spatial Dual-Modality Graph Reasoning method (SDMG-R) to extract key information from unstructured document images. We model document images as dual-modality graphs, nodes of which encode both the visual and textual features of detected text regions, and edges of which represent the spatial relations between neighboring text regions. The key information extraction is solved by iteratively propagating messages along graph edges and reasoning the categories of graph nodes. In order to roundly evaluate our proposed method as well as boost the future research, we release a new dataset named WildReceipt, which is collected and annotated tailored for the evaluation of key information extraction from document images of unseen templates in the wild. It contains 25 key information categories, a total of about 69000 text boxes, and is about 2 times larger than the existing public datasets. Extensive experiments validate that all information including visual features, textual features and spatial relations can benefit key information extraction. It has been shown that SDMG-R can effectively extract key information from document images of unseen templates, and obtain new state-of-the-art results on the recent popular benchmark SROIE and our WildReceipt. Our code and dataset will be publicly released.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142580689-18edb4d7-f716-475c-b1c1-e2b934658cee.png"/>
</div>

## Results and models

### WildReceipt

|                                 Method                                 |     Modality     | Macro F1-Score |                                               Download                                               |
| :--------------------------------------------------------------------: | :--------------: | :------------: | :--------------------------------------------------------------------------------------------------: |
|   [sdmgr_unet16](/configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py)   | Visual + Textual |     0.890      | [model](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_unet16_60e_wildreceipt/sdmgr_unet16_60e_wildreceipt_20220825_151648-22419f37.pth) \| [log](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_unet16_60e_wildreceipt/20220825_151648.log) |
| [sdmgr_novisual](/configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt.py) |     Textual      |     0.873      | [model](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_novisual_60e_wildreceipt/sdmgr_novisual_60e_wildreceipt_20220831_193317-827649d8.pth) \| [log](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_novisual_60e_wildreceipt/20220831_193317.log) |

### WildReceiptOpenset

|                                Method                                 | Modality | Edge F1-Score | Node Macro F1-Score | Node Micro F1-Score |                                 Download                                 |
| :-------------------------------------------------------------------: | :------: | :-----------: | :-----------------: | :-----------------: | :----------------------------------------------------------------------: |
| [sdmgr_novisual_openset](/configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt-openset.py) | Textual  |     0.792     |        0.931        |        0.940        | [model](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_novisual_60e_wildreceipt-openset/sdmgr_novisual_60e_wildreceipt-openset_20220831_200807-dedf15ec.pth) \| [log](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_novisual_60e_wildreceipt-openset/20220831_200807.log) |

## Citation

```bibtex
@misc{sun2021spatial,
      title={Spatial Dual-Modality Graph Reasoning for Key Information Extraction},
      author={Hongbin Sun and Zhanghui Kuang and Xiaoyu Yue and Chenhao Lin and Wayne Zhang},
      year={2021},
      eprint={2103.14470},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
