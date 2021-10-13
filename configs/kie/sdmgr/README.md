# Spatial Dual-Modality Graph Reasoning for Key Information Extraction

## Introduction

[ALGORITHM]

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

## Results and models

### WildReceipt

|                                 Method                                 |     Modality     | Macro F1-Score |                                                                                            Download                                                                                            |
| :--------------------------------------------------------------------: | :--------------: | :------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   [sdmgr_unet16](/configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py)   | Visual + Textual |     0.888      |  [model](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth) \| [log](https://download.openmmlab.com/mmocr/kie/sdmgr/20210520_132236.log.json)  |
| [sdmgr_novisual](/configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt.py) |     Textual      |     0.870      | [model](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_novisual_60e_wildreceipt_20210517-a44850da.pth) \| [log](https://download.openmmlab.com/mmocr/kie/sdmgr/20210517_205829.log.json) |

### WildReceiptOpenset

| Method | Modality | Edge F1-Score | Node Macro F1-Score | Node Micro F1-Score | Download |
| :-------: | :----------: | :--------: | :--------: | :--------: | :--------: |
| [sdmgr_novisual](/configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt_openset.py) |     Textual      |   0.786 | 0.926  | 0.935 | [model](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_novisual_60e_wildreceipt_openset_20210917-d236b3ea.pth) \| [log](https://download.openmmlab.com/mmocr/kie/sdmgr/20210917_050824.log.json) |


:::{note}
1. In the case of openset, the number of node categories is unknown or unfixed, and more node category can be added.
2. To show that our method can handle openset problem, we modify the ground truth of `WildReceipt` to `WildReceiptOpenset`. The `nodes` are just classified into 4 classes: `background, key, value, others`, while adding `edge` labels for each box.
3. The model is used to predict whether two nodes are a pair connecting by a valid edge.

:::
