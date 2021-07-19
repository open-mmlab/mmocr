## 文本测试演示

<div align="center">
    <img src="https://github.com/open-mmlab/mmocr/raw/main/demo/resources/demo_text_recog_pred.jpg" width="200px" alt/><br>

</div>

### 单图演示

我们提供了一个演示脚本，它使用单个 GPU 对[一张图片](/demo/demo_text_recog.jpg)进行文本识别。

```shell
python demo/image_demo.py ${TEST_IMG} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SAVE_PATH} [--imshow] [--device ${GPU_ID}]
```

*模型准备：*
预训练好的模型可以从 [这里](https://mmocr.readthedocs.io/enWe also provide live demos from a webcam as in [mmdetection](https://github.com/open-mmlab/mmdetection/blob/a616886bf1e8de325e6906b8c76b6a4924ef5520/docs/1_exist_data_model.md).
```shell
python demo/image_demo.py demo/demo_text_recog.jpg configs/textrecog/sar/sar_r31_parallel_decoder_academic.py https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth demo/demo_text_recog_pred.jpg
```

预测结果会被保存至 `demo/demo_text_recog_pred.jpg`.


### 多图演示

我们同样提供另一个脚本，它用单个 GPU 对多张图进行批量推断：
```shell
python demo/batch_image_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SAVE_PATH} --images ${IMAGE1} ${IMAGE2} [--imshow] [--device ${GPU_ID}]
```

例如:

```shell
python demo/image_demo.py configs/textrecog/sar/sar_r31_parallel_decoder_academic.py https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth save_results --images demo/demo_text_recog.jpg demo/demo_text_recog.jpg
```

预测结果会被保存至目录 `save_results`.


### 实时识别

我们甚至又提供了使用摄像头实时识别文字的演示，虽然还是不知道有什么用。（[mmdetection](https://github.com/open-mmlab/mmdetection/blob/a616886bf1e8de325e6906b8c76b6a4924ef5520/docs/1_exist_data_model.md) 也这么干了）

```shell
python demo/webcam_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--camera-id ${CAMERA-ID}] \
    [--score-thr ${SCORE_THR}]
```

例如：

```shell
python demo/webcam_demo.py \
    configs/textrecog/sar/sar_r31_parallel_decoder_academic.py \
    https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth
```

### 额外说明

1. 如若打开 `--imshow`，脚本会调用 OpenCV 直接显示出结果图片。
2. 脚本 `image_demo.py` 目前仅支持 GPU， 因此 `--device` 暂不能接受 cpu 为参数。
