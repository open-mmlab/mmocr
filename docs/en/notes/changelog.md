# Changelog of v1.x

## v1.0.0rc3 (03/11/2022)

### Highlights

1. We release several pretrained models using [oCLIP-ResNet](<>) as the backbone, which is a ResNet variant trained with [oCLIP](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880282.pdf) and can significantly boost the performance of text detection models.

2. Preparing datasets is troublesome and tedious, especially in OCR domain where multiple datasets are usually required. In order to free users from laborious work, we designed a [Dataset Preparer](https://mmocr.readthedocs.io/en/dev-1.x/user_guides/data_prepare/dataset_preparer.html), which helps users get a bunch of datasets ready for use, with only **one line of command**!  Dataset Preparer consists of a series of reusable modules responsible for handling different standardized phases throughout the entire preparation process, shortening the development cycle on supporting new datasets.

### New Features & Enhancements

- Add Dataset Preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1484

### Docs

- Update install.md by @rogachevai in https://github.com/open-mmlab/mmocr/pull/1494
- Refine some docs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1455
- Update some dataset preparer related docs by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1502

### Bug Fixes

- Fix offline_eval error caused by new data flow by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1500

## v1.0.0rc2 (14/10/2022)

This release relaxes the version requirement of `MMEngine` to `>=0.1.0, < 1.0.0`.

## v1.0.0rc1 (9/10/2022)

### Highlights

This release fixes a severe bug leading to inaccurate metric report in multi-GPU training.
We release the weights for all the text recognition models in MMOCR 1.0 architecture. The inference shorthand for them are also added back to `ocr.py`. Besides, more documentation chapters are available now.

### New Features & Enhancements

- Simplify the Mask R-CNN config by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1391
- auto scale lr by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1326
- Update paths to pretrain weights by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1416
- Streamline duplicated split_result in pan_postprocessor by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1418
- Update model links in ocr.py and inference.md by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1431
- Update rec configs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1417
- Visualizer refine by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1411
- Support get flops and parameters in dev-1.x by @vansin in https://github.com/open-mmlab/mmocr/pull/1414

### Docs

- intersphinx and api by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1367
- Fix quickrun by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1374
- Fix some docs issues by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1385
- Add Documents for DataElements by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1381
- config english by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1372
- Metrics by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1399
- Add version switcher to menu by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1407
- Data Transforms by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1392
- Fix inference docs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1415
- Fix some docs by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1410
- Add maintenance plan to migration guide by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1413
- Update Recog Models by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1402

### Bug Fixes

- clear metric.results only done in main process by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1379
- Fix a bug in MMDetWrapper by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1393
- Fix browse_dataset.py by @Mountchicken in https://github.com/open-mmlab/mmocr/pull/1398
- ImgAugWrapper: Do not cilp polygons if not applicable by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1231
- Fix CI by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1365
- Fix merge stage test by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1370
- Del CI support for torch 1.5.1 by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1371
- Test windows cu111 by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1373
- Fix windows CI by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1387
- Upgrade pre commit hooks by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1429
- Skip invalid augmented polygons in ImgAugWrapper by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1434

### New Contributors

- @vansin made their first contribution in https://github.com/open-mmlab/mmocr/pull/1414

**Full Changelog**: https://github.com/open-mmlab/mmocr/compare/v1.0.0rc0...v1.0.0rc1

## v1.0.0rc0 (1/9/2022)

We are excited to announce the release of MMOCR 1.0.0rc0.
MMOCR 1.0.0rc0 is the first version of MMOCR 1.x, a part of the OpenMMLab 2.0 projects.
Built upon the new [training engine](https://github.com/open-mmlab/mmengine),
MMOCR 1.x unifies the interfaces of dataset, models, evaluation, and visualization with faster training and testing speed.

### Highlights

1. **New engines**. MMOCR 1.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a general and powerful runner that allows more flexible customizations and significantly simplifies the entrypoints of high-level interfaces.

2. **Unified interfaces**. As a part of the OpenMMLab 2.0 projects, MMOCR 1.x unifies and refactors the interfaces and internal logics of train, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logics to allow the emergence of multi-task/modality algorithms.

3. **Cross project calling**. Benefiting from the unified design, you can use the models implemented in other OpenMMLab projects, such as MMDet. We provide an example of how to use MMDetection's Mask R-CNN through `MMDetWrapper`. Check our documents for more details. More wrappers will be released in the future.

4. **Stronger visualization**. We provide a series of useful tools which are mostly based on brand-new visualizers. As a result, it is more convenient for the users to explore the models and datasets now.

5. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmocr.readthedocs.io/en/dev-1.x/).

### Breaking Changes

We briefly list the major breaking changes here.
We will update the [migration guide](../migration.md) to provide complete details and migration instructions.

#### Dependencies

- MMOCR 1.x relies on MMEngine to run. MMEngine is a new foundational library for training deep learning models in OpenMMLab 2.0 models. The dependencies of file IO and training are migrated from MMCV 1.x to MMEngine.
- MMOCR 1.x relies on MMCV>=2.0.0rc0. Although MMCV no longer maintains the training functionalities since 2.0.0rc0, MMOCR 1.x relies on the data transforms, CUDA operators, and image processing interfaces in MMCV. Note that the package `mmcv` is the version that provide pre-built CUDA operators and `mmcv-lite` does not since MMCV 2.0.0rc0, while `mmcv-full` has been deprecated.

#### Training and testing

- MMOCR 1.x uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of dataset, model, evaluation, and visualizer. Therefore, MMOCR 1.x no longer maintains the building logics of those modules in `mmocr.train.apis` and `tools/train.py`. Those code have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). Please refer to the [migration guide of Runner in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for more details.
- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic as that in training scripts to build the runner.
- The execution points of hooks in the new Runner have been enriched to allow more flexible customization. Please refer to the [migration guide of Hook in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/hook.html) for more details.
- Learning rate and momentum scheduling has been migrated from `Hook` to `Parameter Scheduler` in MMEngine. Please refer to the [migration guide of Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html) for more details.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structures to ease the understanding of the components in runner. Users can read the [config example of MMOCR](../user_guides/config.md) or refer to the [migration guide in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for migration details.
- The file names of configs and models are also refactored to follow the new rules unified across OpenMMLab 2.0 projects. Please refer to the [user guides of config](../user_guides/config.md) for more details.

#### Dataset

The Dataset classes implemented in MMOCR 1.x all inherits from the `BaseDetDataset`, which inherits from the [BaseDataset in MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html). There are several changes of Dataset in MMOCR 1.x.

- All the datasets support to serialize the data list to reduce the memory when multiple workers are built to accelerate data loading.
- The interfaces are changed accordingly.

#### Data Transforms

The data transforms in MMOCR 1.x all inherits from those in MMCV>=2.0.0rc0, which follows a new convention in OpenMMLab 2.0 projects.
The changes are listed as below:

- The interfaces are also changed. Please refer to the [API Reference](https://mmocr.readthedocs.io/en/dev-1.x/)
- The functionality of some data transforms (e.g., `Resize`) are decomposed into several transforms.
- The same data transforms in different OpenMMLab 2.0 libraries have the same augmentation implementation and the logic of the same arguments, i.e., `Resize` in MMDet 3.x and MMOCR 1.x will resize the image in the exact same manner given the same arguments.

#### Model

The models in MMOCR 1.x all inherits from `BaseModel` in MMEngine, which defines a new convention of models in OpenMMLab 2.0 projects. Users can refer to the [tutorial of model](https://mmengine.readthedocs.io/en/latest/tutorials/model.html) in MMengine for more details. Accordingly, there are several changes as the following:

- The model interfaces, including the input and output formats, are significantly simplified and unified following the new convention in MMOCR 1.x. Specifically, all the input data in training and testing are packed into `inputs` and `data_samples`, where `inputs` contains model inputs like a list of image tensors, and `data_samples` contains other information of the current data sample such as ground truths and model predictions. In this way, different tasks in MMOCR 1.x can share the same input arguments, which makes the models more general and suitable for multi-task learning.
- The model has a data preprocessor module, which is used to pre-process the input data of model. In MMOCR 1.x, the data preprocessor usually does necessary steps to form the input images into a batch, such as padding. It can also serve as a place for some special data augmentations or more efficient data transformations like normalization.
- The internal logic of model have been changed. In MMOCR 0.x, model used `forward_train` and `simple_test` to deal with different model forward logics. In MMOCR 1.x and OpenMMLab 2.0, the forward function has three modes: `loss`, `predict`, and `tensor` for training, inference, and tracing or other purposes, respectively. The forward function calls `self.loss()`, `self.predict()`, and `self._forward()` given the modes `loss`, `predict`, and `tensor`, respectively.

#### Evaluation

MMOCR 1.x mainly implements corresponding metrics for each task, which are manipulated by [Evaluator](https://mmengine.readthedocs.io/en/latest/design/evaluator.html) to complete the evaluation.
In addition, users can build evaluator in MMOCR 1.x to conduct offline evaluation, i.e., evaluate predictions that may not produced by MMOCR, prediction follows our dataset conventions. More details can be find in the [Evaluation Tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html) in MMEngine.

#### Visualization

The functions of visualization in MMOCR 1.x are removed. Instead, in OpenMMLab 2.0 projects, we use [Visualizer](https://mmengine.readthedocs.io/en/latest/design/visualization.html) to visualize data. MMOCR 1.x implements `TextDetLocalVisualizer`, `TextRecogLocalVisualizer`, and `KIELocalVisualizer` to allow visualization of ground truths, model predictions, and feature maps, etc., at any place, for the three tasks supported in MMOCR. It also supports to dump the visualization data to any external visualization backends such as Tensorboard and Wandb. Check our [Visualization Document](https://mmocr.readthedocs.io/en/dev-1.x/user_guides/visualization.html) for more details.

### Improvements

- Most models enjoy a performance improvement from the new framework and refactor of data transforms. For example, in MMOCR 1.x, DBNet-R50 achieves **0.854** hmean score on ICDAR 2015, while the counterpart can only get **0.840** hmean score in MMOCR 0.x.
- Support mixed precision training of most of the models. However, the [rest models](https://mmocr.readthedocs.io/en/dev-1.x/user_guides/train_test.html#mixed-precision-training) are not supported yet because the operators they used might not be representable in fp16. We will update the documentation and list the results of mixed precision training.

### Ongoing changes

1. Test-time augmentation: which was supported in MMOCR 0.x, is not implemented yet in this version due to limited time slot. We will support it in the following releases with a new and simplified design.
2. Inference interfaces: a unified inference interfaces will be supported in the future to ease the use of released models.
3. Interfaces of useful tools that can be used in notebook: more useful tools that implemented in the `tools/` directory will have their python interfaces so that they can be used through notebook and in downstream libraries.
4. Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMOCR 1.x.
