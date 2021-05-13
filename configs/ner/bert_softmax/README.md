# Chinese Named Entity Recognition using BERT + Softmax.

## Introduction

[ALGORITHM]
```bibtex
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

## Dataset

### Train Dataset

|  trainset  | text_num | entity_num |
| :--------: | :----------: | :--------: |
| CLUENER2020 |     10748     |     23338     |

### Test Dataset

|  testset  | text_num | entity_num |
| :--------: | :----------: | :--------: |
| CLUENER2020 |     1343     |     2982     |


## Results and models

|                                 Method                                 |  Precision  |   Recall  |  F1-Score |                Download                 |
| :--------------------------------------------------------------------: |:-----------:| :--------:| :-------: | :-------------------------------------: |
|   [bert_softmax](/configs/ner/bert_softmax/bert_softmax_cluener_18e.py)|  0.7793     |    0.7920 |  0.7856   |  [model](https://download.openmmlab.com/mmocr/ner/bert_softmax/bert_softmax_cluener_18e-d96b4a2c.pth) \| [log](https://download.openmmlab.com/mmocr/ner/bert_softmax/20210506_215252.log.json) |
