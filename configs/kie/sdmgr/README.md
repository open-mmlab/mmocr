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

|                                 Method                                 |     Modality     | Macro F1-Score |                                                                Download                                                                 |
| :--------------------------------------------------------------------: | :--------------: | :------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
|   [sdmgr_unet16](/configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py)   | Visual + Textual |     0.876      | [model](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_unet16_60e_wildreceipt_20210405-16a47642.pth) \| [log](https://download.openmmlab.com/mmocr/kie/sdmgr/20210405_104508.log.json) |
| [sdmgr_novisual](/configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt.py) |     Textual      |     0.864      | [model](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_novisual_60e_wildreceipt_20210405-07bc26ad.pth) \| [log](https://download.openmmlab.com/mmocr/kie/sdmgr/20210405_141138.log.json) |
