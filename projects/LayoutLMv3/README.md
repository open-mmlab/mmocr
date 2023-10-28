# LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking

<div>
<a href="https://arxiv.org/abs/2204.08387">[arXiv paper]</a>
</div>

## Description

This is an implementation of [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) based on [MMOCR](https://github.com/open-mmlab/mmocr/tree/dev-1.x), [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine) and [Transformers](https://github.com/huggingface/transformers).

**LayoutLMv3**  Self-supervised pre-training techniques have achieved remarkable progress in Document AI. Most multimodal pre-trained models use a masked language modeling objective to learn bidirectional representations on the text modality, but they differ in pre-training objectives for the image modality. This discrepancy adds difficulty to multimodal representation learning. In this paper, we propose LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-patch alignment objective to learn cross-modal alignment by predicting whether the corresponding image patch of a text word is masked. The simple unified architecture and training objectives make LayoutLMv3 a general-purpose pre-trained model for both text-centric and image-centric Document AI tasks. Experimental results show that LayoutLMv3 achieves state-of-the-art performance not only in text-centric tasks, including form understanding, receipt understanding, and document visual question answering, but also in image-centric tasks such as document image classification and document layout analysis.The code and models are publicly available at https://aka.ms/layoutlmv3.

<center>
<img src="https://user-images.githubusercontent.com/34083603/249024787-fc7b3ea2-2ee8-465c-8cd0-ce0d606f3a87.png">
</center>

## Usage

<!-- > For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Prerequisites

- Python 3.7
- PyTorch 1.6 or higher
- [Transformers](https://github.com/huggingface/transformers) 4.31.0.dev0 or higher
- [MIM](https://github.com/open-mmlab/mim)
- [MMOCR](https://github.com/open-mmlab/mmocr)

### Preparing xfund dataset

In MMOCR's root directory, run the following command to prepare xfund dataset:

```shell
sh projects/LayoutLMv3/scripts/prepare_dataset.sh
```

### Downloading Pre-training LayoutLMv3 model

Download the [LayoutLMv3 Chinese pre-trained model](https://huggingface.co/microsoft/layoutlmv3-base-chinese) from huggingface.

### Training commands

Modify the path of the parameter `hf_pretrained_model` in the config file(`projects/LayoutLMv3/configs/ser/layoutlmv3_1k_xfund_zh_1xbs8.py`)

In MMOCR's root directory, run the following command to train the model:

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
mim train mmocr projects/LayoutLMv3/configs/ser/layoutlmv3_1k_xfund_zh_1xbs8.py --work-dir work_dirs/
```

<!-- To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```bash
mim train mmocr configs/dbnet_dummy-resnet_fpnc_1200e_icdar2015.py --work-dir work_dirs/dummy_mae/ --launcher pytorch --gpus 8
``` -->

### Testing commands

In MMOCR's root directory, run the following command to test the model:

```bash
mim test mmocr projects/LayoutLMv3/configs/ser/layoutlmv3_1k_xfund_zh_1xbs8.py --work-dir work_dirs/ --checkpoint ${CHECKPOINT_PATH}
```

## Results

<!-- >> List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/README.md#results-and-models)
>
> You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project.

|                              Method                               |  Backbone   | Pretrained Model |  Training set   |    Test set    | #epoch | Test size | Precision | Recall | Hmean  |         Download         |
| :---------------------------------------------------------------: | :---------: | :--------------: | :-------------: | :------------: | :----: | :-------: | :-------: | :----: | :----: | :----------------------: |
| [DBNet_dummy](configs/dbnet_dummy-resnet_fpnc_1200e_icdar2015.py) | DummyResNet |        -         | ICDAR2015 Train | ICDAR2015 Test |  1200  |    736    |  0.8853   | 0.7583 | 0.8169 | [model](<>) \| [log](<>) | -->

## Citation

If you find LayoutLMv3 useful in your research or applications, please cite LayoutLMv3 with the following BibTeX entry.

```bibtex
@inproceedings{huang2022layoutlmv3,
  title={Layoutlmv3: Pre-training for document ai with unified text and image masking},
  author={Huang, Yupan and Lv, Tengchao and Cui, Lei and Lu, Yutong and Wei, Furu},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={4083--4091},
  year={2022}
}
```

## Checklist

Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress.

> The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
>
> OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
>
> Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
>
> A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR.

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    > The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmocr.registry.MODELS` and configurable via a config file.

  - [ ] Basic docstrings & proper citation

    > Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd)

  - [ ] Test-time correctness

    > If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone.

  - [ ] A full README

    > As this template does.

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

    > If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range.

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    > Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmocr/blob/76637a290507f151215d299707c57cea5120976e/mmocr/utils/polygon_utils.py#L80-L96)

  - [ ] Unit tests

    > Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmocr/blob/76637a290507f151215d299707c57cea5120976e/tests/test_utils/test_polygon_utils.py#L97-L106)

  - [ ] Code polishing

    > Refactor your code according to reviewer's comment.

  - [ ] Metafile.yml

    > It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/metafile.yml)

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  > In particular, you may have to refactor this README into a standard one. [Example](/configs/textdet/dbnet/README.md)

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
