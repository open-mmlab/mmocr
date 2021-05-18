# Changelog

## v0.2.0 (18/5/2021)

**Highlights**

1. Support Bert-softmax (NAACL'2019)
2. Support DRRG (CVPR'2020)
3. Support FCENet (CVPR'2021)
4. Add text detection and recognition end-to-end demo.

**New Features**

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

**Bug Fixes**

- Fix duplicated points due to transform for textsnake [#130](https://github.com/open-mmlab/mmocr/pull/130)
- Fix CTC loss NaN [#159](https://github.com/open-mmlab/mmocr/pull/159)
- Fix error raised if result is empty in demo [#144](https://github.com/open-mmlab/mmocr/pull/141)
- Fix results missed if a large number of boxes on one image [#98](https://github.com/open-mmlab/mmocr/pull/98)
- Fix package missed in dockerfile [#109](https://github.com/open-mmlab/mmocr/pull/109)

**Improvements**

- Simplify installation procedure [#188](https://github.com/open-mmlab/mmocr/pull/188)
- Add zh-CN README [#70](https://github.com/open-mmlab/mmocr/pull/70) [#95](https://github.com/open-mmlab/mmocr/pull/95)
- Support windows [#89](https://github.com/open-mmlab/mmocr/pull/89)
- Add Colab [#147](https://github.com/open-mmlab/mmocr/pull/147) [#199](https://github.com/open-mmlab/mmocr/pull/199)
- Add 1-step installation using conda environment [#193](https://github.com/open-mmlab/mmocr/pull/193) [#194](https://github.com/open-mmlab/mmocr/pull/194) [#195](https://github.com/open-mmlab/mmocr/pull/195)


## v0.1.0 (7/4/2021)

**Highlights**

- MMOCR is released.

**Main Features**

- Support text detection, text recognition and the corresponding downstream tasks such as key information extraction.
- For text detection, support both single-step (`PSENet`, `PANet`, `DBNet`, `TextSnake`) and two-step (`MaskRCNN`) methods.
- For text recognition, support CTC-loss based method `CRNN`; Encoder-decoder (with attention) based methods `SAR`, `Robustscanner`; Segmentation based method `SegOCR`; Transformer based method `NRTR`.
- For key information extraction, support GCN based method `SDMG-R`.
- Provide checkpoints and log files for all of the methods above.
