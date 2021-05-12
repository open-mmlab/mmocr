## Text Recognition Demo

### Text Recognition Image Demo


We provide a demo script to test a [single demo image](/demo/demo_text_recog.jpg) for text recognition with a single GPU.

*Text Recognition Model Preparation:*
The pre-trained text recognition model can be downloaded from [model zoo](https://mmocr.readthedocs.io/en/latest/modelzoo.html).
Take [SAR](/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py) as an example:

```shell
python demo/image_demo.py ${TEST_IMG} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SAVE_PATH} [--imshow] [--device ${GPU_ID}]
```

Example:

```shell
python demo/image_demo.py demo/demo_text_recog.jpg configs/textrecog/sar/sar_r31_parallel_decoder_academic.py https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth demo/demo_text_recog_pred.jpg
```

The predicted result will be saved as `demo/demo_text_recog_pred.jpg`.

### Text Recognition Webcam Demo

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
    configs/textrecog/sar/sar_r31_parallel_decoder_academic.py \
    https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth
```

### Remarks

1. If `--imshow` is specified, the demo will also show the image with OpenCV.
2. The `image_demo.py` script only supports GPU and so the `--device` parameter cannot take cpu as an argument.
