# Donut

## Description

This is an reimplementation of Donut official repo https://github.com/clovaai/donut.

## Usage

### Prerequisites

- Python 3.7
- PyTorch 1.6 or higher
- [MIM](https://github.com/open-mmlab/mim)
- [MMOCR](https://github.com/open-mmlab/mmocr)
- transformers 4.25.1

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `Donut/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
# Linux
export PYTHONPATH=`pwd`:$PYTHONPATH
# Windows PowerShell
$env:PYTHONPATH=Get-Location
```

### Training commands

In the current directory, run the following command to train the model:

```bash
mim train mmocr configs/donut_cord_30e.py --work-dir work_dirs/donut_cord_30e/
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```bash
mim train mmocr configs/donut_cord_30e.py --work-dir work_dirs/donut_cord_30e/ --launcher pytorch --gpus 8
```

### Testing commands

Before test, you need change tokenizer_cfg in config. The checkpoint shuold be the model save dir, like `work_dirs/donut_cord_30e/`.
In the current directory, run the following command to test the model:

```bash
mim test mmocr configs/donut_cord_30e.py --work-dir work_dirs/donut_cord_30e/ --checkpoint ${CHECKPOINT_PATH}
```

## Results

> List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/README.md#results-and-models)
>
> You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project.

|                 Method                  |     Pretrained Model      | Training set  |   Test set   | #epoch | Test size | TED Acc |   F1   |         Download         |
| :-------------------------------------: | :-----------------------: | :-----------: | :----------: | :----: | :-------: | :-----: | :----: | :----------------------: |
| [Donut_CORD](configs/donut_cord_30e.py) | naver-clova-ix/donut-base | cord-v2 Train | cord-v2 Test |   30   |    736    | 0.8977  | 0.8279 | [model](<>) \| [log](<>) |

## Citation

<!--- cslint:disable -->

```bibtex
@article{Kim_Hong_Yim_Nam_Park_Yim_Hwang_Yun_Han_Park_2021,
title={OCR-free Document Understanding Transformer},
DOI={10.48550/arxiv.2111.15664},
author={Kim, Geewook and Hong, Teakgyu and Yim, Moonbin and Nam, Jeongyeon and Park, Jinyoung and Yim, Jinyeong and Hwang, Wonseok and Yun, Sangdoo and Han, Dongyoon and Park, Seunghyun},
year={2021},
month={Nov},
language={en-US}
}
```

<!--- cslint:enable -->

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

  - [x] Basic docstrings & proper citation

    > Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd)

  - [ ] Test-time correctness

    > If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone.

  - [x] A full README

    > As this template does.

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

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
