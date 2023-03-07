# Changelog of v1.x

## v1.0.0rc5 (07/03/2023)

### Highlights

1. Two new models, ABCNet v2 (inference only) and SPTS are added to `projects/` folder.
2. Announcing `Inferencer`, a unified inference interface in OpenMMLab for everyone's easy access and quick inference with all the pre-trained weights. [Docs](https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html)
3. Users can use test-time augmentation for text recognition tasks. [Docs](https://mmocr.readthedocs.io/en/dev-1.x/user_guides/train_test.html#test-time-augmentation)
4. Support [batch augmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf) through [`BatchAugSampler`](https://github.com/open-mmlab/mmocr/pull/1757), which is a technique used in SPTS.
5. Dataset Preparer has been refactored to allow more flexible configurations. Besides, users are now able to prepare text recognition datasets in LMDB formats. [Docs](qq.com)
6. Some textspotting datasets have been revised to enhance the correctness and consistency with the common practice.
7. Potential spurious warnings from `shapely` have been eliminated.

### Dependency

This version requires MMEngine >= 0.6.0, MMCV >= 2.0.0rc4 and MMDet >= 3.0.0rc5.

### New Features & Enhancements

- Discard deprecated lmdb dataset format and only support img+label now by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1681
- abcnetv2 inference by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1657
- Add RepeatAugSampler by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1678
- SPTS by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1696
- Refactor Inferencers by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1608
- Dynamic return type for rescale_polygons by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1702
- Revise upstream version limit by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1703
- TextRecogCropConverter add crop with opencv warpPersepective function by @KevinNuNu in https://github.com/open-mmlab/mmocr/pull/1667
- change cudnn benchmark to false by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1705
- Add ST-pretrained DB-series models and logs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1635
- Only keep meta and state_dict when publish model by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1729
- Rec TTA by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1401
- Speedup formatting by replacing np.transpose with torch… by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1719
- Support auto import modules from registry. by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1731
- Support batch visualization & dumping in Inferencer by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1722
- add a new argument font_properties to set a specific font file in order to draw Chinese characters properly by @KevinNuNu in https://github.com/open-mmlab/mmocr/pull/1709
- Refactor data converter and gather by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1707
- Support batch augmentation through BatchAugSampler by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1757
- Put all registry into registry.py by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1760
- train by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1756
- configs for regression benchmark by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1755

### Docs

- update the link of DBNet by @AllentDan in https://github.com/open-mmlab/mmocr/pull/1672
- Add notice for default branch switching by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1693
- docs: Add twitter discord medium youtube link by @vansin in https://github.com/open-mmlab/mmocr/pull/1724
- Remove unsupported datasets in docs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1670

### Bug Fixes

- Update dockerfile by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1671
- Explicitly create np object array for compatibility by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1691
- Fix a minor error in docstring by @Mountchicken in https://github.com/open-mmlab/mmocr/pull/1685
- Fix lint by @triple-Mu in https://github.com/open-mmlab/mmocr/pull/1694
- Fix LoadOCRAnnotation ut by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1695
- Fix isort pre-commit error by @KevinNuNu in https://github.com/open-mmlab/mmocr/pull/1697
- Update owners by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1699
- Detect intersection before using shapley.intersection to eliminate spurious warnings by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1710
- Fix some inferencer bugs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1706
- Fix textocr ignore flag by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1712
- Add missing softmax in ASTER forward_test by @Mountchicken in https://github.com/open-mmlab/mmocr/pull/1718
- Fix head in readme by @vansin in https://github.com/open-mmlab/mmocr/pull/1727
- Fix some browse dataset script bugs and draw textdet gt instance with ignore flags by @KevinNuNu in https://github.com/open-mmlab/mmocr/pull/1701
- icdar textrecog ann parser skip data with ignore flag by @KevinNuNu in https://github.com/open-mmlab/mmocr/pull/1708
- bezier_to_polygon ->  bezier2polygon by @double22a in https://github.com/open-mmlab/mmocr/pull/1739
- Fix docs recog CharMetric P/R error definition by @KevinNuNu in https://github.com/open-mmlab/mmocr/pull/1740
- Remove outdated resources in demo/ by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1747
- Fix wrong ic13 textspotting split data; add lexicons to ic13, ic15 and totaltext by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1758
- SPTS readme by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1761

### New Contributors

- @triple-Mu made their first contribution in https://github.com/open-mmlab/mmocr/pull/1694
- @double22a made their first contribution in https://github.com/open-mmlab/mmocr/pull/1739

**Full Changelog**: https://github.com/open-mmlab/mmocr/compare/v1.0.0rc5...v1.0.0rc6

## v1.0.0rc5 (06/01/2023)

### Highlights

1. Two models, Aster and SVTR, are added to our model zoo. The full implementation of ABCNet is also available now.
2. Dataset Preparer supports 5 more datasets: CocoTextV2, FUNSD, TextOCR, NAF, SROIE.
3. We have 4 more text recognition transforms, and two helper transforms. See https://github.com/open-mmlab/mmocr/pull/1646 https://github.com/open-mmlab/mmocr/pull/1632 https://github.com/open-mmlab/mmocr/pull/1645 for details.
4. The transform, `FixInvalidPolygon`, is getting smarter at dealing with invalid polygons, and now capable of handling more weird annotations. As a result, a complete training cycle on TotalText dataset can be performed bug-free. The weights of DBNet and FCENet pretrained on TotalText are also released.

### New Features & Enhancements

- Update ic15 det config according to DataPrepare by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1617
- Refactor icdardataset metainfo to lowercase. by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1620
- Add ASTER Encoder by @Mountchicken in https://github.com/open-mmlab/mmocr/pull/1239
- Add ASTER decoder by @Mountchicken in https://github.com/open-mmlab/mmocr/pull/1625
- Add ASTER config by @Mountchicken in https://github.com/open-mmlab/mmocr/pull/1238
- Update ASTER config by @Mountchicken in https://github.com/open-mmlab/mmocr/pull/1629
- Support browse_dataset.py to visualize original dataset by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1503
- Add CocoTextv2 to dataset preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1514
- Add Funsd to dataset preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1550
- Add TextOCR to Dataset Preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1543
- Refine example projects and readme by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1628
- Enhance FixInvalidPolygon, add RemoveIgnored transform by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1632
- ConditionApply by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1646
- Add NAF to dataset preparer by @Mountchicken in https://github.com/open-mmlab/mmocr/pull/1609
- Add SROIE to dataset preparer by @FerryHuang in https://github.com/open-mmlab/mmocr/pull/1639
- Add svtr decoder by @willpat1213 in https://github.com/open-mmlab/mmocr/pull/1448
- Add missing unit tests by @Mountchicken in https://github.com/open-mmlab/mmocr/pull/1651
- Add svtr encoder by @willpat1213 in https://github.com/open-mmlab/mmocr/pull/1483
- ABCNet train by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1610
- Totaltext cfgs for DB and FCE by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1633
- Add Aliases to models by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1611
- SVTR transforms by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1645
- Add SVTR framework and configs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1621
- Issue Template by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1663

### Docs

- Add Chinese translation for browse_dataset.py by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1647
- updata abcnet doc by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1658
- update the dbnetpp\`s readme file by @zhuyue66 in https://github.com/open-mmlab/mmocr/pull/1626
- Inferencer docs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1744

### Bug Fixes

- nn.SmoothL1Loss beta can not be zero in PyTorch 1.13 version by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1616
- ctc loss bug if target is empty by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1618
- Add torch 1.13 by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1619
- Remove outdated tutorial link by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1627
- Dev 1.x  some doc mistakes by @KevinNuNu in https://github.com/open-mmlab/mmocr/pull/1630
- Support custom font to visualize some languages (e.g. Korean) by @ProtossDragoon in https://github.com/open-mmlab/mmocr/pull/1567
- db_module_loss，negative number encountered in sqrt by @KevinNuNu in https://github.com/open-mmlab/mmocr/pull/1640
- Use int instead of np.int by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1636
- Remove support for py3.6 by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1660

### New Contributors

- @zhuyue66 made their first contribution in https://github.com/open-mmlab/mmocr/pull/1626
- @KevinNuNu made their first contribution in https://github.com/open-mmlab/mmocr/pull/1630
- @FerryHuang made their first contribution in https://github.com/open-mmlab/mmocr/pull/1639
- @willpat1213 made their first contribution in https://github.com/open-mmlab/mmocr/pull/1448

**Full Changelog**: https://github.com/open-mmlab/mmocr/compare/v1.0.0rc4...v1.0.0rc5

## v1.0.0rc4 (06/12/2022)

### Highlights

1. Dataset Preparer can automatically generate base dataset configs at the end of the preparation process, and supports 6 more datasets: IIIT5k, CUTE80, ICDAR2013, ICDAR2015, SVT, SVTP.
2. Introducing our `projects/` folder - implementing new models and features into OpenMMLab's algorithm libraries has long been complained to be troublesome due to the rigorous requirements on code quality, which could hinder the fast iteration of SOTA models and might discourage community members from sharing their latest outcome here. We now introduce `projects/` folder, where some experimental features, frameworks and models can be placed, only needed to satisfy the minimum requirement on the code quality. Everyone is welcome to post their implementation of any great ideas in this folder! We also add the first [example project](https://github.com/open-mmlab/mmocr/tree/dev-1.x/projects/example_project) to illustrate what we expect a good project to have (check out the raw content of README.md for more info!).
3. Inside the `projects/` folder, we are releasing the preview version of ABCNet, which is the first implementation of text spotting models in MMOCR. It's inference-only now, but the full implementation will be available very soon.

### New Features & Enhancements

- Add SVT to dataset preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1521
- Polish bbox2poly by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1532
- Add SVTP to dataset preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1523
- Iiit5k converter by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1530
- Add cute80 to dataset preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1522
- Add IC13 preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1531
- Add 'Projects/' folder, and the first example project by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1524
- Rename to {dataset-name}\_task_train/test by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1541
- Add print_config.py to the tools by @IncludeMathH in https://github.com/open-mmlab/mmocr/pull/1547
- Add get_md5 by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1553
- Add config generator by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1552
- Support IC15_1811  by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1556
- Update CT80 config by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1555
- Add config generators to all textdet and textrecog configs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1560
- Refactor TPS by @Mountchicken in https://github.com/open-mmlab/mmocr/pull/1240
- Add TextSpottingConfigGenerator by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1561
- Add common typing by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1596
- Update textrecog config and readme  by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1597
- Support head loss or postprocessor is None for only infer by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1594
- Textspotting datasample by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1593
- Simplify mono_gather by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1588
- ABCNet v1 infer by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1598

### Docs

- Add Chinese Guidance on How to Add New Datasets to Dataset Preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1506
- Update the qq group link by @vansin in https://github.com/open-mmlab/mmocr/pull/1569
- Collapse some sections; update logo url by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1571
- Update dataset preparer (CN) by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1591

### Bug Fixes

- Fix two bugs in dataset preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1513
- Register bug of CLIPResNet by @jyshee in https://github.com/open-mmlab/mmocr/pull/1517
- Being more conservative on Dataset Preparer by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1520
- python -m pip upgrade in windows by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1525
- Fix wildreceipt metafile by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1528
- Fix Dataset Preparer Extract by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1527
- Fix ICDARTxtParser by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1529
- Fix Dataset Zoo Script by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1533
- Fix crop without padding and recog metainfo delete unuse info by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1526
- Automatically create nonexistent directory for base  configs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1535
- Change mmcv.dump to mmengine.dump by @ProtossDragoon in https://github.com/open-mmlab/mmocr/pull/1540
- mmocr.utils.typing -> mmocr.utils.typing_utils by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1538
- Wildreceipt tests by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1546
- Fix judge exist dir by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1542
- Fix IC13 textdet config by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1563
- Fix IC13 textrecog annotations by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1568
- Auto scale lr by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1584
- Fix icdar data parse for text containing separator by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1587
- Fix textspotting ut by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1599
- Fix TextSpottingConfigGenerator and TextSpottingDataConverter by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1604
- Keep E2E Inferencer output simple by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1559

### New Contributors

- @jyshee made their first contribution in https://github.com/open-mmlab/mmocr/pull/1517
- @ProtossDragoon made their first contribution in https://github.com/open-mmlab/mmocr/pull/1540
- @IncludeMathH made their first contribution in https://github.com/open-mmlab/mmocr/pull/1547

**Full Changelog**: https://github.com/open-mmlab/mmocr/compare/v1.0.0rc3...v1.0.0rc4

## v1.0.0rc3 (03/11/2022)

### Highlights

1. We release several pretrained models using [oCLIP-ResNet](https://github.com/open-mmlab/mmocr/blob/1.x/configs/backbone/oclip/README.md) as the backbone, which is a ResNet variant trained with [oCLIP](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880282.pdf) and can significantly boost the performance of text detection models.

2. Preparing datasets is troublesome and tedious, especially in OCR domain where multiple datasets are usually required. In order to free our users from laborious work, we designed a [Dataset Preparer](https://mmocr.readthedocs.io/en/dev-1.x/user_guides/data_prepare/dataset_preparer.html) to help you get a bunch of datasets ready for use, with only **one line of command**!  Dataset Preparer is also crafted to consist of a series of reusable modules, each responsible for handling one of the standardized phases throughout the preparation process, shortening the development cycle on supporting new datasets.

### New Features & Enhancements

- Add Dataset Preparer by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1484

* support modified resnet structure used in oCLIP by @HannibalAPE in https://github.com/open-mmlab/mmocr/pull/1458
* Add oCLIP configs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1509

### Docs

- Update install.md by @rogachevai in https://github.com/open-mmlab/mmocr/pull/1494
- Refine some docs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1455
- Update some dataset preparer related docs by @xinke-wang in https://github.com/open-mmlab/mmocr/pull/1502
- oclip readme by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/1505

### Bug Fixes

- Fix offline_eval error caused by new data flow by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/1500

### New Contributors

- @rogachevai made their first contribution in https://github.com/open-mmlab/mmocr/pull/1494
- @HannibalAPE made their first contribution in https://github.com/open-mmlab/mmocr/pull/1458

**Full Changelog**: https://github.com/open-mmlab/mmocr/compare/v1.0.0rc2...v1.0.0rc3

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
