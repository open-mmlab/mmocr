# Changelog
## v0.2.0 (16/5/2021)

**Highlights**

- MMOCR is compiling-free via moving textdet postprocessing and ops to mmcv 1.3.4 or later.
- Add a new OCR downstream task-NER.
- Add two new text detection methods: DRRG and FCENet.
- Add end-to-end demo.

## v0.1.0 (7/4/2021)

**Highlights**

- MMOCR is released.

**Main Features**

- Support text detection, text recognition and the corresponding downstream tasks such as key information extraction.
- For text detection, support both single-step (`PSENet`, `PANet`, `DBNet`, `TextSnake`) and two-step (`MaskRCNN`) methods.
- For text recognition, support CTC-loss based method `CRNN`; Encoder-decoder (with attention) based methods `SAR`, `Robustscanner`; Segmentation based method `SegOCR`; Transformer based method `NRTR`.
- For key information extraction, support GCN based method `SDMG-R`.
- Provide checkpoints and log files for all of the methods above.
