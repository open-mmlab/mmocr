## Text Detection Demo

<div align="center">
    <img src="https://github.com/open-mmlab/mmocr/raw/main/demo/resources/demo_text_det_pred.jpg"/><br>

</div>

### Text Detection Single Image Demo

We provide a demo script to test a [single image](/demo/demo_text_det.jpg) for text detection with a single GPU.

*Text Detection Model Preparation:*
The pre-trained text detection model can be downloaded from [model zoo](https://mmocr.readthedocs.io/en/latest/modelzoo.html).
Take [PANet](/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) as an example:

```shell
python demo/image_demo.py ${TEST_IMG} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SAVE_PATH} [--imshow] [--device ${GPU_ID}]
```

Example:

```shell
python demo/image_demo.py demo/demo_text_det.jpg configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth demo/demo_text_det_pred.jpg
```

The predicted result will be saved as `demo/demo_text_det_pred.jpg`.

### Text Detection Multiple Image Demo

We provide a demo script to test multi-images in batch mode for text detection with a single GPU.

*Text Detection Model Preparation:*
The pre-trained text detection model can be downloaded from [model zoo](https://mmocr.readthedocs.io/en/latest/modelzoo.html).
Take [PANet](/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) as an example:

```shell
python demo/batch_image_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SAVE_PATH} --images ${IMAGE1} ${IMAGE2} [--imshow] [--device ${GPU_ID}]
```

Example:

```shell
python demo/batch_image_demo.py configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth save_results --images demo/demo_text_det.jpg demo/demo_text_det.jpg
```

The predicted result will be saved in folder `save_results`.

### Text Detection Webcam Demo

We also provide live demos from a webcam as in [mmdetection](https://github.com/open-mmlab/mmdetection/blob/a616886bf1e8de325e6906b8c76b6a4924ef5520/docs/1_exist_data_model.md).

```shell
python demo/webcam_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--camera-id ${CAMERA-ID}] \
    [--score-thr ${SCORE_THR}]
```

Examples:

```shell
python demo/webcam_demo.py \
    configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py \ https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth
```

### Remarks

1. If `--imshow` is specified, the demo will also show the image with OpenCV.
2. The `image_demo.py` script only supports GPU and so the `--device` parameter cannot take cpu as an argument.
