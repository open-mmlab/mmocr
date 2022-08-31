# Useful Tools

## Analysis Tools

### Dataset Visualization Tool "browse_datasets.py"

MMOCR provides a dataset visualization tool `tools/analysis_tools/browse_datasets.py` to help users troubleshoot possible dataset-related problems. You just need to specify the path to the training config and the tool will automatically plots the images transformed by corresponding data pipelines with the GT labels. The following example demonstrates how to use the tool to visualize the training data used by the "DBNet_R50_icdar2015" model.

```Bash
# Example: Visualizing the training data used by dbnet_r50dcn_v2_fpnc_1200e_icadr2015
python tools/analysis_tools/browse_dataset.py configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py
```

The visualization results will be like:

<div align="center">

![browse dataset](https://aicarrier.feishu.cn/space/api/box/stream/download/asynccode/?code=N2E3NzhlY2Q2YmY3OGMzODA4M2IxZWNkY2Q0ZjcyMGVfNWViVUlaUnBraVV3NkJ4dkRLSmRhSDVyZVVnY1dmUEpfVG9rZW46Ym94Y25rNzdCUHNtV1M5Z1hSYUVRRW14OGNiXzE2NjE4NDQzODU6MTY2MTg0Nzk4NV9WNA)

</div>

Based on this tool, users can easily verify if the annotation of a custom dataset is correct. Also, you can verify if the data augmentation strategies are running as you expected by modifying `train_pipeline` in the configuration file. The optional parameters of `browse_dataset.py` are as follows.

|                 |       |                                                                                       |
| --------------- | ----- | ------------------------------------------------------------------------------------- |
| ARGS            | Type  | Description                                                                           |
| config          | str   | (required) Path to the config.                                                        |
| --output-dir    | str   | If GUI is not available, specifying an output path to save the visualization results. |
| --show-interval | float | Interval of visualization (s), defaults to 2.                                         |

### Offline Evaluation Tool "offline_eval.py"

For saved prediction results, we provide an offline evaluation script `tools/analysis_tools/offline_eval.py`. The following example demonstrates how to use this tool to evaluate the output of the "PSENet" model offline.

```Bash
# When running the test script for the first time, you can save the output of the model by specifying the --save-preds parameter
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --save-preds
# Example: Testing on PSENet
python tools/test.py configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py epoch_600.pth --save-preds

# Then, using the saved outputs for offline evaluation
python tools/analysis_tool/offline_eval.py ${CONFIG_FILE} ${PRED_FILE}
# Example: Offline evaluation of saved PSENet results
python tools/analysis_tools/offline_eval.py configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py work_dirs/psenet_r50_fpnf_600e_icdar2015/epoch_600.pth_predictions.pkl
```

`-save-preds` saves the output to `work_dir/CONFIG_NAME/MODEL_NAME_predictions.pkl` by default

In addition, based on this tool, users can also convert predictions obtained from other libraries into MMOCR-supported formats, then use MMOCR's built-in metrics to evaluate them.

|               |       |                                   |
| ------------- | ----- | --------------------------------- |
| ARGS          | Type  | Description                       |
| config        | str   | (required) Path to the config.    |
| pkl_results   | str   | (required) The saved predictions. |
| --cfg-options | float | Override configs. [Example](<>)   |
