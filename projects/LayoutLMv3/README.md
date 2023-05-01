# Dummy ResNet Wrapper

> This is a README template for community `projects/`.

> All the fields in this README are **mandatory** for others to understand what you have achieved in this implementation. If you still feel unclear about the requirements, please read our [contribution guide](https://mmocr.readthedocs.io/en/dev-1.x/notes/contribution_guide.html), [projects FAQ](../faq.md), or approach us in [Discussions](https://github.com/open-mmlab/mmocr/discussions).

## Description

> Share any information you would like others to know. For example:
>
> Author: @xxx.
>
> This is an implementation of \[XXX\].

This project implements a dummy ResNet wrapper, which literally does nothing new but prints "hello world" during initialization.

## Usage

> For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`.

### Prerequisites

- Python 3.7
- PyTorch 1.6 or higher
- [MIM](https://github.com/open-mmlab/mim)
- [MMOCR](https://github.com/open-mmlab/mmocr)

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `example_project/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
# Linux
export PYTHONPATH=`pwd`:$PYTHONPATH
# Windows PowerShell
$env:PYTHONPATH=Get-Location
```

### Training commands

In MMOCR's root directory, run the following command to train the model:

```bash
mim train mmocr configs/dbnet_dummy-resnet_fpnc_1200e_icdar2015.py --work-dir work_dirs/dummy_mae/
```

To train on multiple GPUs, e.g. 8 GPUs, run the following command:

```bash
mim train mmocr configs/dbnet_dummy-resnet_fpnc_1200e_icdar2015.py --work-dir work_dirs/dummy_mae/ --launcher pytorch --gpus 8
```

### Testing commands

In MMOCR's root directory, run the following command to test the model:

```bash
mim test mmocr configs/dbnet_dummy-resnet_fpnc_1200e_icdar2015.py --work-dir work_dirs/dummy_mae/ --checkpoint ${CHECKPOINT_PATH}
```

## Results

> List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/README.md#results-and-models)
>
> You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project.

|                              Method                               |  Backbone   | Pretrained Model |  Training set   |    Test set    | #epoch | Test size | Precision | Recall | Hmean  |         Download         |
| :---------------------------------------------------------------: | :---------: | :--------------: | :-------------: | :------------: | :----: | :-------: | :-------: | :----: | :----: | :----------------------: |
| [DBNet_dummy](configs/dbnet_dummy-resnet_fpnc_1200e_icdar2015.py) | DummyResNet |        -         | ICDAR2015 Train | ICDAR2015 Test |  1200  |    736    |  0.8853   | 0.7583 | 0.8169 | [model](<>) \| [log](<>) |

## Citation

> You may remove this section if not applicable.

```bibtex
@software{MMOCR_Contributors_OpenMMLab_Text_Detection_2020,
author = {{MMOCR Contributors}},
license = {Apache-2.0},
month = {8},
title = {{OpenMMLab Text Detection, Recognition and Understanding Toolbox}},
url = {https://github.com/open-mmlab/mmocr},
version = {0.3.0},
year = {2020}
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

  - [ ] Finish the code

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
