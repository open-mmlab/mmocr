# Spatial Dual-Modality Graph Reasoning for Key Information Extraction

## Abstract

<!-- [ABSTRACT] -->

Key information extraction from document images is of paramount importance in office automation. Conventional template matching based approaches fail to generalize well to document images of unseen templates, and are not robust against text recognition errors. In this paper, we propose an end-to-end Spatial Dual-Modality Graph Reasoning method (SDMG-R) to extract key information from unstructured document images. We model document images as dual-modality graphs, nodes of which encode both the visual and textual features of detected text regions, and edges of which represent the spatial relations between neighboring text regions. The key information extraction is solved by iteratively propagating messages along graph edges and reasoning the categories of graph nodes. In order to roundly evaluate our proposed method as well as boost the future research, we release a new dataset named WildReceipt, which is collected and annotated tailored for the evaluation of key information extraction from document images of unseen templates in the wild. It contains 25 key information categories, a total of about 69000 text boxes, and is about 2 times larger than the existing public datasets. Extensive experiments validate that all information including visual features, textual features and spatial relations can benefit key information extraction. It has been shown that SDMG-R can effectively extract key information from document images of unseen templates, and obtain new state-of-the-art results on the recent popular benchmark SROIE and our WildReceipt. Our code and dataset will be publicly released.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142580689-18edb4d7-f716-475c-b1c1-e2b934658cee.png"/>
</div>


## Citation

<!-- [ALGORITHM] -->

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

:::{note}
1. For `sdmgr_novisual`, images are not needed for training and testing. So fake `img_prefix` can be used in configs. As well, fake `file_name` can be used in annotation files.
:::

### WildReceiptOpenset

| Method | Modality | Edge F1-Score | Node Macro F1-Score | Node Micro F1-Score | Download |
| :-------: | :----------: | :--------: | :--------: | :--------: | :--------: |
| [sdmgr_novisual](/configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt_openset.py) |     Textual      |   0.786 | 0.926  | 0.935 | [model](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_novisual_60e_wildreceipt_openset_20210917-d236b3ea.pth) \| [log](https://download.openmmlab.com/mmocr/kie/sdmgr/20210917_050824.log.json) |


:::{note}
1. In the case of openset, the number of node categories is unknown or unfixed, and more node category can be added.
2. To show that our method can handle openset problem, we modify the ground truth of `WildReceipt` to `WildReceiptOpenset`. The `nodes` are just classified into 4 classes: `background, key, value, others`, while adding `edge` labels for each box.
3. The model is used to predict whether two nodes are a pair connecting by a valid edge.
4. You can learn more about the key differences between CloseSet and OpenSet annotations in our [tutorial](tutorials/kie_closeset_openset.md).
:::
