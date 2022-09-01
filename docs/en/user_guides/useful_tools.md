# Useful Tools

## Analysis Tools

### Dataset Visualization Tool

MMOCR provides a dataset visualization tool `tools/analysis_tools/browse_datasets.py` to help users troubleshoot possible dataset-related problems. You just need to specify the path to the training config and the tool will automatically plots the images transformed by corresponding data pipelines with the GT labels. The following example demonstrates how to use the tool to visualize the training data used by the "DBNet_R50_icdar2015" model.

```Bash
# Example: Visualizing the training data used by dbnet_r50dcn_v2_fpnc_1200e_icadr2015
python tools/analysis_tools/browse_dataset.py configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py
```

The visualization results will be like:

<center class="half">
    <img src="https://user-images.githubusercontent.com/24622904/187611542-01e9aa94-fc12-4756-964b-a0e472522a3a.jpg" width="250"/><img src="https://user-images.githubusercontent.com/24622904/187611555-3f5ea616-863d-4538-884f-bccbebc2f7e7.jpg" width="250"/><img src="https://user-images.githubusercontent.com/24622904/187611581-88be3970-fbfe-4f62-8cdf-7a8a7786af29.jpg" width="250"/>
</center>

Based on this tool, users can easily verify if the annotation of a custom dataset is correct. Also, you can verify if the data augmentation strategies are running as you expected by modifying `train_pipeline` in the configuration file. The optional parameters of `browse_dataset.py` are as follows.

| ARGS            | Type  | Description                                                                           |
| --------------- | ----- | ------------------------------------------------------------------------------------- |
| config          | str   | (required) Path to the config.                                                        |
| --output-dir    | str   | If GUI is not available, specifying an output path to save the visualization results. |
| --show-interval | float | Interval of visualization (s), defaults to 2.                                         |

### Offline Evaluation Tool

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

| ARGS          | Type  | Description                       |
| ------------- | ----- | --------------------------------- |
| config        | str   | (required) Path to the config.    |
| pkl_results   | str   | (required) The saved predictions. |
| --cfg-options | float | Override configs. [Example](<>)   |
