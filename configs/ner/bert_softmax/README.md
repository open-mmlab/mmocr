# Bert

> [Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805)

<!-- [ALGORITHM] -->

## Abstract

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142802652-ecc6500d-e5dc-4ffa-98f4-f5b247b9245c.png"/>
</div>

## Dataset

### Train Dataset

|  trainset   | text_num | entity_num |
| :---------: | :------: | :--------: |
| CLUENER2020 |  10748   |   23338    |

### Test Dataset

|   testset   | text_num | entity_num |
| :---------: | :------: | :--------: |
| CLUENER2020 |   1343   |    2982    |

## Results and models

|                          Method                           |                           Pretrain                           | Precision | Recall | F1-Score |                           Download                           |
| :-------------------------------------------------------: | :----------------------------------------------------------: | :-------: | :----: | :------: | :----------------------------------------------------------: |
| [bert_softmax](/configs/ner/bert_softmax/bert_softmax_cluener_18e.py) | [pretrain](https://download.openmmlab.com/mmocr/ner/bert_softmax/bert_pretrain.pth) |  0.7885   | 0.7998 |  0.7941  | [model](https://download.openmmlab.com/mmocr/ner/bert_softmax/bert_softmax_cluener-eea70ea2.pth) \| [log](https://download.openmmlab.com/mmocr/ner/bert_softmax/20210514_172645.log.json) |

## Citation

```bibtex
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```
