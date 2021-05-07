# Chinese Named Entity Recognition using BERT(softmax).

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
|  trainset  | text_num | entity_categories |
| :--------: | :----------: | :--------: |
| CLUENER2020 |     10748     |     10     |

### Test Dataset

|  testset  | text_num | entity_categories |
| :--------: | :----------: | :--------: |
| CLUENER2020 |     1343     |     10     |


## Results and models

|                                 Method                                 |     Precision      |     Recall     |  F1-Score |                                                                                            Download                                                                                            |
| :--------------------------------------------------------------------: | :--------------:  | :--------------: | :------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   [bert_softmax](/configs/ner/bert_softmax/bert_softmax_cluener_18e.py)    | 0.7793 |  0.7920 |  0.7856     |
