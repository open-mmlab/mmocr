# Changelog

## v0.3.0 (25/8/2021)

### Highlights
1. We add a new text recognition model -- SATRN! Its pretrained checkpoint achieves the best performance over other provided text recognition models. A lighter version of SATRN is also released which can obtain ~98% of the performance of the original model with only 45 MB in size. ([@2793145003](https://github.com/2793145003)) [#405](https://github.com/open-mmlab/mmocr/pull/405)
2. Improve the demo script, `ocr.py`, which supports applying end-to-end text detection, text recognition and key information extraction models on images with easy-to-use commands. Users can find its full documentation in the demo section. ([@samayala22](https://github.com/samayala22), [@manjrekarom](https://github.com/manjrekarom)) [#371](https://github.com/open-mmlab/mmocr/pull/371), [#386](https://github.com/open-mmlab/mmocr/pull/386), [#400](https://github.com/open-mmlab/mmocr/pull/400), [#374](https://github.com/open-mmlab/mmocr/pull/374), [#428](https://github.com/open-mmlab/mmocr/pull/428)
3. Our documentation is reorganized into a clearer structure. More useful contents are on the way! [#409](https://github.com/open-mmlab/mmocr/pull/409), [#454](https://github.com/open-mmlab/mmocr/pull/454)
4. The requirement of `Polygon3` is removed since this project is no longer maintained or distributed. We unified all its references to equivalent substitutions in `shapely` instead. [#448](https://github.com/open-mmlab/mmocr/pull/448)

### Breaking Changes & Migration Guide
1. Upgrade version requirement of MMDetection to 2.14.0 to avoid bugs [#382](https://github.com/open-mmlab/mmocr/pull/382)
2. MMOCR now has its own model and layer registries inherited from MMDetection's or MMCV's counterparts. ([#436](https://github.com/open-mmlab/mmocr/pull/436)) The modified hierarchical structure of the model registries are now organized as follows.

```text
mmcv.MODELS -> mmdet.BACKBONES -> BACKBONES
mmcv.MODELS -> mmdet.NECKS -> NECKS
mmcv.MODELS -> mmdet.ROI_EXTRACTORS -> ROI_EXTRACTORS
mmcv.MODELS -> mmdet.HEADS -> HEADS
mmcv.MODELS -> mmdet.LOSSES -> LOSSES
mmcv.MODELS -> mmdet.DETECTORS -> DETECTORS
mmcv.ACTIVATION_LAYERS -> ACTIVATION_LAYERS
mmcv.UPSAMPLE_LAYERS -> UPSAMPLE_LAYERS
```

To migrate your old implementation to our new backend, you need to change the import path of any registries and their corresponding builder functions (including `build_detectors`) from `mmdet.models.builder` to `mmocr.models.builder`. If you have referred to any model or layer of MMDetection or MMCV in your model config, you need to add `mmdet.` or `mmcv.` prefix to its name to inform the model builder of the right namespace to work on.

Interested users may check out [MMCV's tutorial on Registry](https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html) for in-depth explanations on its mechanism.


### New Features
- Automatically replace SyncBN with BN for inference [#420](https://github.com/open-mmlab/mmocr/pull/420), [#453](https://github.com/open-mmlab/mmocr/pull/453)
- Support batch inference for CRNN and SegOCR [#407](https://github.com/open-mmlab/mmocr/pull/407)
- Support exporting documentation in pdf or epub format [#406](https://github.com/open-mmlab/mmocr/pull/406)
- Support `persistent_workers` option in data loader [#459](https://github.com/open-mmlab/mmocr/pull/459)

### Bug Fixes
- Remove depreciated key in kie_test_imgs.py [#381](https://github.com/open-mmlab/mmocr/pull/381)
- Fix dimension mismatch in batch testing/inference of DBNet [#383](https://github.com/open-mmlab/mmocr/pull/383)
- Fix the problem of dice loss which stays at 1 with an empty target given [#408](https://github.com/open-mmlab/mmocr/pull/408)
- Fix a wrong link in ocr.py ([@naarkhoo](https://github.com/naarkhoo)) [#417](https://github.com/open-mmlab/mmocr/pull/417)
- Fix undesired assignment to "pretrained" in test.py [#418](https://github.com/open-mmlab/mmocr/pull/418)
- Fix a problem in polygon generation of DBNet [#421](https://github.com/open-mmlab/mmocr/pull/421), [#443](https://github.com/open-mmlab/mmocr/pull/443)
- Skip invalid annotations in totaltext_converter [#438](https://github.com/open-mmlab/mmocr/pull/438)
- Add zero division handler in poly utils, remove Polygon3 [#448](https://github.com/open-mmlab/mmocr/pull/448)

### Improvements
- Replace lanms-proper with lanms-neo to support installation on Windows (with special thanks to [@gen-ko](https://github.com/gen-ko) who has re-distributed this package!)
- Support MIM [#394](https://github.com/open-mmlab/mmocr/pull/394)
- Add tests for PyTorch 1.9 in CI [#401](https://github.com/open-mmlab/mmocr/pull/401)
- Enables fullscreen layout in readthedocs [#413](https://github.com/open-mmlab/mmocr/pull/413)
- General documentation enhancement [#395](https://github.com/open-mmlab/mmocr/pull/395)
- Update version checker [#427](https://github.com/open-mmlab/mmocr/pull/427)
- Add copyright info [#439](https://github.com/open-mmlab/mmocr/pull/439)
- Update citation information [#440](https://github.com/open-mmlab/mmocr/pull/440)

### Contributors

We thank [@2793145003](https://github.com/2793145003), [@samayala22](https://github.com/samayala22), [@manjrekarom](https://github.com/manjrekarom), [@naarkhoo](https://github.com/naarkhoo), [@gen-ko](https://github.com/gen-ko), [@duanjiaqi](https://github.com/duanjiaqi), [@gaotongxiao](https://github.com/gaotongxiao), [@cuhk-hbsun](https://github.com/cuhk-hbsun), [@innerlee](https://github.com/innerlee), [@wdsd641417025](https://github.com/wdsd641417025) for their contribution to this release!

## v0.2.1 (20/7/2021)

### Highlights
1. Upgrade to use MMCV-full **>= 1.3.8** and MMDetection **>= 2.13.0** for latest features
2. Add ONNX and TensorRT export tool, supporting the deployment of DBNet, PSENet, PANet and CRNN (experimental) [#278](https://github.com/open-mmlab/mmocr/pull/278), [#291](https://github.com/open-mmlab/mmocr/pull/291), [#300](https://github.com/open-mmlab/mmocr/pull/300), [#328](https://github.com/open-mmlab/mmocr/pull/328)
3. Unified parameter initialization method which uses init_cfg in config files [#365](https://github.com/open-mmlab/mmocr/pull/365)

### New Features
- Support TextOCR dataset [#293](https://github.com/open-mmlab/mmocr/pull/293)
- Support Total-Text dataset [#266](https://github.com/open-mmlab/mmocr/pull/266), [#273](https://github.com/open-mmlab/mmocr/pull/273), [#357](https://github.com/open-mmlab/mmocr/pull/357)
- Support grouping text detection box into lines [#290](https://github.com/open-mmlab/mmocr/pull/290), [#304](https://github.com/open-mmlab/mmocr/pull/304)
- Add benchmark_processing script that benchmarks data loading process [#261](https://github.com/open-mmlab/mmocr/pull/261)
- Add SynthText preprocessor for text recognition models [#351](https://github.com/open-mmlab/mmocr/pull/351), [#361](https://github.com/open-mmlab/mmocr/pull/361)
- Support batch inference during testing [#310](https://github.com/open-mmlab/mmocr/pull/310)
- Add user-friendly OCR inference script [#366](https://github.com/open-mmlab/mmocr/pull/366)

### Bug Fixes

- Fix improper class ignorance in SDMGR Loss [#221](https://github.com/open-mmlab/mmocr/pull/221)
- Fix potential numerical zero division error in DRRG [#224](https://github.com/open-mmlab/mmocr/pull/224)
- Fix installing requirements with pip and mim [#242](https://github.com/open-mmlab/mmocr/pull/242)
- Fix dynamic input error of DBNet [#269](https://github.com/open-mmlab/mmocr/pull/269)
- Fix space parsing error in LineStrParser [#285](https://github.com/open-mmlab/mmocr/pull/285)
- Fix textsnake decode error [#264](https://github.com/open-mmlab/mmocr/pull/264)
- Correct isort setup [#288](https://github.com/open-mmlab/mmocr/pull/288)
- Fix a bug in SDMGR config [#316](https://github.com/open-mmlab/mmocr/pull/316)
- Fix kie_test_img for KIE nonvisual [#319](https://github.com/open-mmlab/mmocr/pull/319)
- Fix metafiles [#342](https://github.com/open-mmlab/mmocr/pull/342)
- Fix different device problem in FCENet [#334](https://github.com/open-mmlab/mmocr/pull/334)
- Ignore improper tailing empty characters in annotation files [#358](https://github.com/open-mmlab/mmocr/pull/358)
- Docs fixes [#247](https://github.com/open-mmlab/mmocr/pull/247), [#255](https://github.com/open-mmlab/mmocr/pull/255), [#265](https://github.com/open-mmlab/mmocr/pull/265), [#267](https://github.com/open-mmlab/mmocr/pull/267), [#268](https://github.com/open-mmlab/mmocr/pull/268), [#270](https://github.com/open-mmlab/mmocr/pull/270), [#276](https://github.com/open-mmlab/mmocr/pull/276), [#287](https://github.com/open-mmlab/mmocr/pull/287), [#330](https://github.com/open-mmlab/mmocr/pull/330), [#355](https://github.com/open-mmlab/mmocr/pull/355), [#367](https://github.com/open-mmlab/mmocr/pull/367)
- Fix NRTR config [#356](https://github.com/open-mmlab/mmocr/pull/356), [#370](https://github.com/open-mmlab/mmocr/pull/370)

### Improvements
- Add backend for resizeocr [#244](https://github.com/open-mmlab/mmocr/pull/244)
- Skip image processing pipelines in SDMGR novisual [#260](https://github.com/open-mmlab/mmocr/pull/260)
- Speedup DBNet [#263](https://github.com/open-mmlab/mmocr/pull/263)
- Update mmcv installation method in workflow [#323](https://github.com/open-mmlab/mmocr/pull/323)
- Add part of Chinese documentations [#353](https://github.com/open-mmlab/mmocr/pull/353), [#362](https://github.com/open-mmlab/mmocr/pull/362)
- Add support for ConcatDataset with two workflows [#348](https://github.com/open-mmlab/mmocr/pull/348)
- Add list_from_file and list_to_file utils [#226](https://github.com/open-mmlab/mmocr/pull/226)
- Speed up sort_vertex [#239](https://github.com/open-mmlab/mmocr/pull/239)
- Support distributed evaluation of KIE [#234](https://github.com/open-mmlab/mmocr/pull/234)
- Add pretrained FCENet on IC15 [#258](https://github.com/open-mmlab/mmocr/pull/258)
- Support CPU for OCR demo [#227](https://github.com/open-mmlab/mmocr/pull/227)
- Avoid extra image pre-processing steps [#375](https://github.com/open-mmlab/mmocr/pull/375)


## v0.2.0 (18/5/2021)

### Highlights

1. Add the NER approach Bert-softmax (NAACL'2019)
2. Add the text detection method DRRG (CVPR'2020)
3. Add the text detection method FCENet (CVPR'2021)
4. Increase the ease of use via adding text detection and recognition end-to-end demo, and colab online demo.
5. Simplify the installation.

### New Features

- Add Bert-softmax for Ner task [#148](https://github.com/open-mmlab/mmocr/pull/148)
- Add DRRG [#189](https://github.com/open-mmlab/mmocr/pull/189)
- Add FCENet [#133](https://github.com/open-mmlab/mmocr/pull/133)
- Add end-to-end demo [#105](https://github.com/open-mmlab/mmocr/pull/105)
- Support batch inference [#86](https://github.com/open-mmlab/mmocr/pull/86) [#87](https://github.com/open-mmlab/mmocr/pull/87) [#178](https://github.com/open-mmlab/mmocr/pull/178)
- Add TPS preprocessor for text recognition [#117](https://github.com/open-mmlab/mmocr/pull/117) [#135](https://github.com/open-mmlab/mmocr/pull/135)
- Add demo documentation [#151](https://github.com/open-mmlab/mmocr/pull/151) [#166](https://github.com/open-mmlab/mmocr/pull/166) [#168](https://github.com/open-mmlab/mmocr/pull/168) [#170](https://github.com/open-mmlab/mmocr/pull/170) [#171](https://github.com/open-mmlab/mmocr/pull/171)
- Add checkpoint for Chinese recognition [#156](https://github.com/open-mmlab/mmocr/pull/156)
- Add metafile [#175](https://github.com/open-mmlab/mmocr/pull/175) [#176](https://github.com/open-mmlab/mmocr/pull/176) [#177](https://github.com/open-mmlab/mmocr/pull/177) [#182](https://github.com/open-mmlab/mmocr/pull/182) [#183](https://github.com/open-mmlab/mmocr/pull/183)
- Add support for numpy array inference [#74](https://github.com/open-mmlab/mmocr/pull/74)

### Bug Fixes

- Fix the duplicated point bug due to transform for textsnake [#130](https://github.com/open-mmlab/mmocr/pull/130)
- Fix CTC loss NaN [#159](https://github.com/open-mmlab/mmocr/pull/159)
- Fix error raised if result is empty in demo [#144](https://github.com/open-mmlab/mmocr/pull/141)
- Fix results missing if one image has a large number of boxes [#98](https://github.com/open-mmlab/mmocr/pull/98)
- Fix package missing in dockerfile [#109](https://github.com/open-mmlab/mmocr/pull/109)

### Improvements

- Simplify installation procedure via removing compiling [#188](https://github.com/open-mmlab/mmocr/pull/188)
- Speed up panet post processing so that it can detect dense texts [#188](https://github.com/open-mmlab/mmocr/pull/188)
- Add zh-CN README [#70](https://github.com/open-mmlab/mmocr/pull/70) [#95](https://github.com/open-mmlab/mmocr/pull/95)
- Support windows [#89](https://github.com/open-mmlab/mmocr/pull/89)
- Add Colab [#147](https://github.com/open-mmlab/mmocr/pull/147) [#199](https://github.com/open-mmlab/mmocr/pull/199)
- Add 1-step installation using conda environment [#193](https://github.com/open-mmlab/mmocr/pull/193) [#194](https://github.com/open-mmlab/mmocr/pull/194) [#195](https://github.com/open-mmlab/mmocr/pull/195)


## v0.1.0 (7/4/2021)

### Highlights

- MMOCR is released.

### Main Features

- Support text detection, text recognition and the corresponding downstream tasks such as key information extraction.
- For text detection, support both single-step (`PSENet`, `PANet`, `DBNet`, `TextSnake`) and two-step (`MaskRCNN`) methods.
- For text recognition, support CTC-loss based method `CRNN`; Encoder-decoder (with attention) based methods `SAR`, `Robustscanner`; Segmentation based method `SegOCR`; Transformer based method `NRTR`.
- For key information extraction, support GCN based method `SDMG-R`.
- Provide checkpoints and log files for all of the methods above.
