# SPTS: Single-Point Text Spotting

<div>
<a href="https://arxiv.org/abs/2112.07917">[arXiv paper]</a>
</div>

## Description

This is an implementation of [SPTS](https://github.com/shannanyinxiang/SPTS) based on [MMOCR](https://github.com/open-mmlab/mmocr/tree/dev-1.x), [MMCV](https://github.com/open-mmlab/mmcv), and [MMEngine](https://github.com/open-mmlab/mmengine).

Existing scene text spotting (i.e., end-to-end text detection and recognition) methods rely on costly bounding box annotations (e.g., text-line, word-level, or character-level bounding boxes). For the first time, we demonstrate that training scene text spotting models can be achieved with an extremely low-cost annotation of a single-point for each instance. We propose an end-to-end scene text spotting method that tackles scene text spotting as a sequence prediction task. Given an image as input, we formulate the desired detection and recognition results as a sequence of discrete tokens and use an auto-regressive Transformer to predict the sequence. The proposed method is simple yet effective, which can achieve state-of-the-art results on widely used benchmarks. Most significantly, we show that the performance is not very sensitive to the positions of the point annotation, meaning that it can be much easier to be annotated or even be automatically generated than the bounding box that requires precise positions. We believe that such a pioneer attempt indicates a significant opportunity for scene text spotting applications of a much larger scale than previously possible.

<center>
<img src="https://user-images.githubusercontent.com/22607038/215685203-fbf2d00c-39d3-48bb-9d05-4fd28c56431c.png">
</center>

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Prerequisites

- Python 3.7
- PyTorch 1.6 or higher
- [MIM](https://github.com/open-mmlab/mim)
- [MMOCR](https://github.com/open-mmlab/mmocr)

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `SPTS/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
# Linux
export PYTHONPATH=`pwd`:$PYTHONPATH
# Windows PowerShell
$env:PYTHONPATH=Get-Location
```

### Dataset

As of now, the implementation uses datasets provided by SPTS for pre-training, and uses MMOCR's datasets for fine-tuning and testing. It's because the test split of SPTS's datasets does not contain enough information for e2e evaluation; and MMOCR's dataset preparer has not yet supported all the datasets used in SPTS. *We are working on this issue, and they will be available in MMOCR's dataset preparer very soon.*

Please follow these steps to prepare the datasets:

- Download and extract all the SPTS datasets into `spts-data/` following [SPTS official guide](https://github.com/shannanyinxiang/SPTS#dataset).

- Use [Dataset Preparer](https://mmocr.readthedocs.io/en/dev-1.x/user_guides/data_prepare/dataset_preparer.html) to prepare `icdar2013`, `icdar2015` and `totaltext` for `textspotting` task. Then create a soft link to `data/` directory in the project root directory:

  ```shell
  ln -s ../../data/ .
  ```

### Training commands

In the current directory, run the following command to train the model:

#### Pretrain

```bash
mim train mmocr config/spts/spts_resnet50_150e_pretrain-spts.py --work-dir work_dirs/ --amp
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```bash
mim train mmocr config/spts/spts_resnet50_150e_pretrain-spts.py --work-dir work_dirs/ --launcher pytorch --gpus 8 --amp
```

#### Finetune

Similarly, run the following command to finetune the model on a dataset (e.g. icdar2013):

```bash
mim train mmocr config/spts/spts_resnet50_200e_icdar2013.py --work-dir work_dirs/ --cfg-options "load_from={CHECKPOINT_PATH}" --amp
```

To finetune on multiple GPUs, e.g. 8 GPUs, run the following command:

```bash
mim train mmocr config/spts/spts_resnet50_200e_icdar2013.py --work-dir work_dirs/ --launcher pytorch --gpus 8 --cfg-options "load_from={CHECKPOINT_PATH}" --amp
```

### Testing commands

In the current directory, run the following command to test the model on a dataset (e.g. icdar2013):

```bash
mim test mmocr config/spts/spts_resnet50_200e_icdar2013.py --work-dir work_dirs/ --checkpoint ${CHECKPOINT_PATH}
```

## Convert Weights from Official Repo

Users may download the weights from [SPTS](https://github.com/shannanyinxiang/SPTS#inference) and use the conversion script to convert them into MMOCR format.

```bash
python tools/ckpt_adapter.py [SPTS_WEIGHTS_PATH] [MMOCR_WEIGHTS_PATH]
```

## Results

All the models are trained on 8x A100 GPUs with AMP on (`--amp`). The actual batch size is 64.

| Name       | Pretrained                                                                              | Generic | Weak  | Strong | Download                                                                              |
| ---------- | --------------------------------------------------------------------------------------- | ------- | ----- | ------ | ------------------------------------------------------------------------------------- |
| ICDAR 2013 | [model](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_150e_pretrain-spts/spts_resnet50_150e_pretrain-spts-c9fe4c78.pth) / \[log\]([model](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_150e_pretrain-spts/20230223_194550.log) | 87.10   | 91.46 | 93.41  | [model](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_200e_icdar2013/spts_resnet50_200e_icdar2013-64cb4d31.pth) / [log](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_200e_icdar2013/20230303_140316.log) |
| ICDAR 2015 | [model](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_150e_pretrain-spts/spts_resnet50_150e_pretrain-spts-c9fe4c78.pth) / \[log\]([model](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_150e_pretrain-spts/20230223_194550.log) | 69.09   | 73.45 | 79.19  | [model](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_200e_icdar2015/spts_resnet50_200e_icdar2015-d6e8621c.pth) / [log](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_200e_icdar2015/20230302_230026.log) |

|   Name    | Pretrained                                                                             | None-Hmean | Full-Hmean | Download                                                                              |
| :-------: | -------------------------------------------------------------------------------------- | :--------: | :--------: | ------------------------------------------------------------------------------------- |
| Totaltext | [model](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_150e_pretrain-spts/spts_resnet50_150e_pretrain-spts-c9fe4c78.pth) / \[log\]([model](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_150e_pretrain-spts/20230223_194550.log) |   73.99    |   82.34    | [model](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_200e_totaltext/spts_resnet50_200e_totaltext-e3521af6.pth) / [log](https://download.openmmlab.com/mmocr/textspotting/spts/spts_resnet50_200e_totaltext/20230303_103040.log) |

## Citation

If you find SPTS useful in your research or applications, please cite SPTS with the following BibTeX entry.

```BibTeX
@inproceedings{peng2022spts,
  title={SPTS: Single-Point Text Spotting},
  author={Peng, Dezhi and Wang, Xinyu and Liu, Yuliang and Zhang, Jiaxin and Huang, Mingxin and Lai, Songxuan and Zhu, Shenggao and Li, Jing and Lin, Dahua and Shen, Chunhua and Bai, Xiang and Jin, Lianwen},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.

OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.

Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.

A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmocr.registry.MODELS` and configurable via a config file. -->

  - [x] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [x] A full README

    <!-- As this template does. -->

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmocr/blob/76637a290507f151215d299707c57cea5120976e/mmocr/utils/polygon_utils.py#L80-L96) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmocr/blob/76637a290507f151215d299707c57cea5120976e/tests/test_utils/test_polygon_utils.py#L97-L106) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](/configs/textdet/dbnet/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
