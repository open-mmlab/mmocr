## OCR End2End Demo

<div align="center">
    <img src="../../resources/demo_ocr_pred.jpg"/><br>

</div>

### End-to-End Test Image Demo

To end-to-end test a single image with text detection and recognition simutaneously:

```shell
python demo/ocr_image_demo.py demo/demo_text_det.jpg demo/output.jpg
```

- The default config for text detection and recognition are [PSENet_ICDAR2015](../../configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py) and [SAR](../../configs/textrecog/sar/sar_r31_parallel_decoder_academic.py), respectively.

- The predicted result will be saved as `demo/output.jpg`.
- To use other algorithms of text detection and recognition, please set arguments: `--det-config`, `--det-ckpt`, `--recog-config`, `--recog-ckpt`.

### Remarks

1. If `--imshow` is specified, the demo will also show the image with OpenCV.
2. The `ocr_image_demo.py` script only supports GPU and so the `--device` parameter cannot take cpu as an argument.
