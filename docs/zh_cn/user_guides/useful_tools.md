# 常用工具

## 分析工具

### 数据集可视化工具

MMOCR 提供了数据集可视化工具 `tools/analysis_tools/browse_datasets.py` 以辅助用户排查可能遇到的数据集相关的问题。用户只需要指定所使用的训练配置文件路径，该工具即可自动将经过数据流水线（data pipeline）处理过的图像及其对应的真实标签绘制出来。例如，以下命令演示了如何使用该工具对 "DBNet_R50_icdar2015" 模型使用的训练数据进行可视化操作：

```Bash
# 示例：可视化 dbnet_r50dcn_v2_fpnc_1200e_icadr2015 使用的训练数据
python tools/analysis_tools/browse_dataset.py configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py
```

效果如下图所示：

<center class="half">
    <img src="https://user-images.githubusercontent.com/24622904/187611542-01e9aa94-fc12-4756-964b-a0e472522a3a.jpg" width="250"/><img src="https://user-images.githubusercontent.com/24622904/187611555-3f5ea616-863d-4538-884f-bccbebc2f7e7.jpg" width="250"/><img src="https://user-images.githubusercontent.com/24622904/187611581-88be3970-fbfe-4f62-8cdf-7a8a7786af29.jpg" width="250"/>
</center>

基于此工具，用户可以方便地验证自定义数据集的标注格式是否正确；也可以通过修改配置文件中的 `train_pipeline` 来验证不同的数据增强策略组合是否符合自己的预期。`browse_dataset.py` 的可选参数如下：

| 参数            | 类型  | 说明                                                                                                     |
| --------------- | ----- | -------------------------------------------------------------------------------------------------------- |
| config          | str   | （必须）配置文件路径。                                                                                   |
| --output-dir    | str   | 可视化结果保存路径。对于不存在图形界面的设备，如服务器集群等，用户可以通过指定输出路径来保存可视化结果。 |
| --show-interval | float | 可视化图像间隔秒数，默认为 2。                                                                           |

### 离线评测工具

对于已保存的预测结果，我们提供了离线评测脚本 `tools/analysis_tools/offline_eval.py`。例如，以下代码演示了如何使用该工具对 "PSENet" 模型的输出结果进行离线评估：

```Bash
# 初次运行测试脚本时，用户可以通过指定 --save-preds 参数来保存模型的输出结果
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --save-preds
# 示例：对 PSENet 进行测试
python tools/test.py configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py epoch_600.pth --save-preds

# 之后即可使用已保存的输出文件进行离线评估
python tools/analysis_tool/offline_eval.py ${CONFIG_FILE} ${PRED_FILE}
# 示例：对已保存的 PSENet 结果进行离线评估
python tools/analysis_tools/offline_eval.py configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py work_dirs/psenet_r50_fpnf_600e_icdar2015/epoch_600.pth_predictions.pkl
```

`--save-preds` 默认将输出结果保存至 `work_dir/CONFIG_NAME/MODEL_NAME_predictions.pkl`

此外，基于此工具，用户也可以将其他算法库获取的预测结果转换成 MMOCR 支持的格式，从而使用 MMOCR 内置的评估指标来对其他算法库的模型进行评测。

| 参数          | 类型  | 说明                                     |
| ------------- | ----- | ---------------------------------------- |
| config        | str   | （必须）配置文件路径。                   |
| pkl_results   | str   | （必须）预先保存的预测结果文件。         |
| --cfg-options | float | 用于覆写配置文件中的指定参数。[示例](<>) |
