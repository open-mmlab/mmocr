# 数据准备

## 一键式数据准备脚本

MMOCR 提供了统一的一站式数据集准备脚本 `prepare_dataset.py`。

仅需一行命令即可完成数据的下载、解压，以及格式转换。

```bash
python tools/dataset_converters/prepare_dataset.py [$DATASET_NAME] --task [$TASK] --nproc [$NPROC]
```

| 参数         | 类型 | 说明                                                                                                  |
| ------------ | ---- | ----------------------------------------------------------------------------------------------------- |
| dataset_name | str  | （必须）需要准备的数据集名称。                                                                        |
| --task       | str  | 将数据集格式转换为指定任务的 MMOCR 格式。可选项为： 'textdet', 'textrecog', 'textspotting' 和 'kie'。 |
| --nproc      | str  | 使用的进程数，默认为 4。                                                                              |

例如，以下命令展示了如何使用该脚本为 ICDAR2015 数据集准备文本检测任务所需的数据。

```bash
python tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet
```

该脚本也支持同时准备多个数据集，例如，以下命令展示了如何使用该脚本同时为 ICDAR2015 和 TotalText 数据集准备文本识别任务所需的数据。

```bash
python tools/dataset_converters/prepare_dataset.py icdar2015 totaltext --task textrecog
```

下表展示了目前支持一键下载及格式转换的数据集。

| 数据集名称  | 文本检测任务 | 文本识别任务 | 端到端文本检测识别任务 | 关键信息抽取任务 |
| ----------- | ------------ | ------------ | ---------------------- | ---------------- |
| icdar2015   | ✓            | ✓            | ✓                      |                  |
| totaltext   | ✓            | ✓            | ✓                      |                  |
| wildreceipt | ✓            | ✓            | ✓                      | ✓                |

## 进阶用法\[待更新\]
