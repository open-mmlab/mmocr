## 文本检测演示

<div align="center">
    <img src="https://github.com/open-mmlab/mmocr/raw/main/demo/resources/demo_text_det_pred.jpg"/><br>

</div>


### 单图演示

我们提供了一个演示脚本，它使用单个 GPU 对[一张图片](/demo/demo_text_det.jpg)进行文本检测。

```shell
python demo/image_demo.py ${TEST_IMG} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SAVE_PATH} [--imshow] [--device ${GPU_ID}]
```

*模型准备：*
预训练好的模型可以从 [这里](https://mmocr.readthedocs.io/en/latest/modelzoo.html) 下载。以 [PANet](/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) 为例：


```shell
python demo/image_demo.py demo/demo_text_det.jpg configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth demo/demo_text_det_pred.jpg
```

预测结果将会保存到 `demo/demo_text_det_pred.jpg`。


### 多图演示

我们同样提供另一个脚本，它用单个 GPU 对多张图进行批量推断：

```shell
python demo/batch_image_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SAVE_PATH} --images ${IMAGE1} ${IMAGE2} [--imshow] [--device ${GPU_ID}]
```

同样以 [PANet](/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) 为例：

```shell
python demo/batch_image_demo.py configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth save_results --images demo/demo_text_det.jpg demo/demo_text_det.jpg
```

预测结果会被保存至目录 `save_results`。


### 实时检测

我们甚至还提供了使用摄像头实时检测文字的演示，虽然不知道有什么用。（[mmdetection](https://github.com/open-mmlab/mmdetection/blob/a616886bf1e8de325e6906b8c76b6a4924ef5520/docs/1_exist_data_model.md) 也这么干了）

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
    configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py \ https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth
```

### 额外说明

1. 如若打开 `--imshow`，脚本会调用 OpenCV 直接显示出结果图片。
2. 脚本 `image_demo.py` 目前仅支持 GPU， 因此 `--device` 暂不能接受 cpu 为参数。
