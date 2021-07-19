## OCR 端对端演示

<div align="center">
    <img src="https://github.com/open-mmlab/mmocr/raw/main/demo/resources/demo_ocr_pred.jpg"/><br>
</div>

### 端对端测试图像演示

运行以下命令，可以同时对测试图像进行文本检测和识别：

```shell
python demo/ocr_image_demo.py demo/demo_text_det.jpg demo/output.jpg
```

- 我们默认使用 [PSENet_ICDAR2015](/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py) 作为文本检测配置，默认文本识别配置则为 [SAR](/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py)。

- 测试结果会保存到 `demo/output.jpg`。
- 如果想尝试其他模型，请使用 `--det-config`, `--det-ckpt`, `--recog-config`, `--recog-ckpt` 参数设置配置及模型文件。
- 设置 `--batch-mode`, `--batch-size` 以对图片进行批量测试。

### 额外说明

1. 如若打开 `--imshow`，脚本会调用 OpenCV 直接显示出结果图片。
2. 该脚本 (`ocr_image_demo.py`) 目前仅支持 GPU， 因此 `--device` 暂不能接受 cpu 为参数。
3. （实验性功能）如若打开 `--ocr-in-lines`，在同一行上的 OCR 检测框会被自动合并并输出。
