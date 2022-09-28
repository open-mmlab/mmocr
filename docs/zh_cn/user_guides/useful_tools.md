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

### 计算flops和参数量

安装依赖

```shell
pip install fvcore
```

获取 `dbnet_resnet18_fpnc_100k_synthtext.py` flops 和参数量的示例命令。

```shell
python tools/analysis_tools/get_flops.py configs/textdet/dbnet/dbnet_resnet18_fpnc_100k_synthtext.py --shape 1024 1024
```

输出如下：

```shell
input shape is  (1, 3, 1024, 1024)
| module                       | #parameters or shape   | #flops     |
|:-----------------------------|:-----------------------|:-----------|
| model                        | 12.341M                | 63.955G    |
|  backbone                    |  11.177M               |  38.159G   |
|   backbone.conv1             |   9.408K               |   2.466G   |
|    backbone.conv1.weight     |    (64, 3, 7, 7)       |            |
|   backbone.bn1               |   0.128K               |   83.886M  |
|    backbone.bn1.weight       |    (64,)               |            |
|    backbone.bn1.bias         |    (64,)               |            |
|   backbone.layer1            |   0.148M               |   9.748G   |
|    backbone.layer1.0         |    73.984K             |    4.874G  |
|    backbone.layer1.1         |    73.984K             |    4.874G  |
|   backbone.layer2            |   0.526M               |   8.642G   |
|    backbone.layer2.0         |    0.23M               |    3.79G   |
|    backbone.layer2.1         |    0.295M              |    4.853G  |
|   backbone.layer3            |   2.1M                 |   8.616G   |
|    backbone.layer3.0         |    0.919M              |    3.774G  |
|    backbone.layer3.1         |    1.181M              |    4.842G  |
|   backbone.layer4            |   8.394M               |   8.603G   |
|    backbone.layer4.0         |    3.673M              |    3.766G  |
|    backbone.layer4.1         |    4.721M              |    4.837G  |
|  neck                        |  0.836M                |  14.887G   |
|   neck.lateral_convs         |   0.246M               |   2.013G   |
|    neck.lateral_convs.0.conv |    16.384K             |    1.074G  |
|    neck.lateral_convs.1.conv |    32.768K             |    0.537G  |
|    neck.lateral_convs.2.conv |    65.536K             |    0.268G  |
|    neck.lateral_convs.3.conv |    0.131M              |    0.134G  |
|   neck.smooth_convs          |   0.59M                |   12.835G  |
|    neck.smooth_convs.0.conv  |    0.147M              |    9.664G  |
|    neck.smooth_convs.1.conv  |    0.147M              |    2.416G  |
|    neck.smooth_convs.2.conv  |    0.147M              |    0.604G  |
|    neck.smooth_convs.3.conv  |    0.147M              |    0.151G  |
|  det_head                    |  0.329M                |  10.909G   |
|   det_head.binarize          |   0.164M               |   10.909G  |
|    det_head.binarize.0       |    0.147M              |    9.664G  |
|    det_head.binarize.1       |    0.128K              |    20.972M |
|    det_head.binarize.3       |    16.448K             |    1.074G  |
|    det_head.binarize.4       |    0.128K              |    83.886M |
|    det_head.binarize.6       |    0.257K              |    67.109M |
|   det_head.threshold         |   0.164M               |            |
|    det_head.threshold.0      |    0.147M              |            |
|    det_head.threshold.1      |    0.128K              |            |
|    det_head.threshold.3      |    16.448K             |            |
|    det_head.threshold.4      |    0.128K              |            |
|    det_head.threshold.6      |    0.257K              |            |
!!!Please be cautious if you use the results in papers. You may need to check if all ops are supported and verify that the flops computation is correct.
```
