# 代码结构变动

MMOCR 为了兼顾文本检测、识别和关键信息提取等任务，在初版设计时存在许多欠缺考虑的地方。在本次 1.0 版本的升级中，MMOCR 同步提出了新的模型架构，旨在尽量与 OpenMMLab 整体的设计对齐，且在算法库内部达成结构上的统一。虽然本次升级并非完全后向兼容，但所有的变动都是有迹可循的。因此，我们在本章节总结出了开发者可能会关心的改动，供有需要的用户参考。

## 整体改动

MMOCR 0.x 存在着对模块功能边界定义不清晰的问题。在 MMOCR 1.0 中，我们重构了模型模块的设计，并定义了它们的模块边界。

- 考虑到方向差异过大，MMOCR 1.0 中取消了对命名实体识别的支持。

- 模型中计算损失（loss）的部分模块被抽象化为 Module Loss，转换原始标注为损失目标（loss target）的功能也被包括在内。另一个模块抽象 Postprocessor 则负责在预测时解码模型原始输出为对应任务的 `DataSample`。

- 所有模型的输入简化为包含图像原始特征的 `inputs` 和图片元信息的 `List[DataSample]`。输出格式也得到统一，训练时是包含 loss 的字典，测试时的输出为包含预测结果的对应任务的 [`DataSample`](<>)。

- Module Loss 来源于 0.x 版本中实现与单个模型强相关的 `XXLoss` 类，它们在 1.0 中均被统一重命名为`XXModuleLoss`的形式（如`DBLoss` 被重命名为 `DBModuleLoss`）, `head` 传入的 loss 配置参数名也从 `loss` 改为 `module_loss`。

- 与模型实现无关的通用损失类名称保持 `XXLoss` 的形式，并放置于 `mmocr/models/common/losses` 下，如 [`MaskedBCELoss`](mmocr.models.common.losses.MaskedBCELoss)。

- `mmocr/models/common/losses` 下的改动：0.x 中 `DiceLoss` 被重名为 [`MaskedDiceLoss`](mmocr.models.common.losses.MaskedDiceLoss)。`FocalLoss` 被移除。

- 增加了起源于 label converter 的 Dictionary 模块，它会在文本识别和关键信息提取任务中被用到。

## 文本检测

### 关键改动（太长不看版）

- 旧版的模型权重仍然适用于新版，但需要将权重字典 `state_dict` 中以 `bbox_head` 开头的字段重命名为 `det_head`。

- 计算 target 有关的变换 `XXTargets` 被转移到了 `XXModuleLoss` 中。

### SingleStageTextDetector

- 原本继承链为 `mmdet.BaseDetector->SingleStageDetector->SingleStageTextDetector`，现在改为直接继承自 `BaseDetector`, 中间的 `SingleStageDetector` 被删除。

- `bbox_head` 改名为 `det_head`。

- `train_cfg`、`test_cfg`和`pretrained`字段被移除。

- `forward_train()` 与 `simple_test()` 分别被重构为 `loss()` 与 `predict()` 方法。其中 `simple_test()` 中负责将模型原始输出拆分并输入 `head.get_bounary()` 的部分被整合进了 `BaseTextDetPostProcessor` 中。

- `TextDetectorMixin` 中只实现了 `show_result()`方法，实现与 `TextDetLocalVisualizer` 重合，因此已经被移除。

### Head

- `HeadMixin` 为`XXXHead` 在 0.x 版本中必须继承的基类，现在被 `BaseTextDetHead` 代替。里面的 `get_boundary()` 和 `resize_boundary()` 方法被重写为 `BaseTextDetPostProcessor` 的 `__call__()` 和 `rescale()` 方法。

### ModuleLoss

- 文本检测中特有的数据变换 `XXXTargets` 全部移动到 `XXXModuleLoss._get_target_single` 中，与生成 target 相关的配置不再在数据流水线（pipeline）中设置，转而在 `XXXLoss` 中被配置。例如，`DBNetTargets` 的实现被移动到 `DBModuleLoss._get_target_single()`中，而用户可以通过设置 `DBModuleLoss` 的初始化参数来控制损失目标的生成。

### Postprocessor

- 原本的 `XXXPostprocessor.__call__()` 中的逻辑转移到重构后的 `XXXPostprocessor.get_text_instances()` 。

- `BasePostprocessor` 重构为 `BaseTextDetPostProcessor`，此基类会将模型输出的预测结果拆分并逐个进行处理，并支持根据 `scale_factor` 自动缩放输出的多边形（polygon）或界定框（bounding box）。

## 文本识别

### 关键改动（太长不看版）

- 由于字典序发生了变化，且存在部分模型架构上的 bug 被修复，旧版的识别模型权重已经不再能直接应用于 1.0 中，我们将会在后续为有需要的用户推出迁移脚本教程。

- 0.x 版本中的 SegOCR 支持暂时移除，TPS-CRNN 会在后续版本中被支持。

- 测试时增强（test time augmentation）在此版本中暂未支持，但将会在后续版本中更新。

- Label converter 模块被移除，里面的功能被拆分至 Dictionary, ModuleLoss 和 Postprocessor 模块中。

- 统一模型中对 `max_seq_len` 的定义为模型的原始输出长度。

### Label Converter

- 原有的 label converter 存在拼写错误 (label convertor)，我们通过删除掉这个类规避了这个问题。

- 负责对字符/字符串与数字索引互相转换的部分被提取至 [`Dictionary`](mmocr.models.common.Dictionary) 类中。

- 在旧版本中，不同的 label converter 会有不一样的特殊字符集和字符序。在 0.x 版本中，字符序如下：

  | Converter                       | 字符序                                    |
  | ------------------------------- | ----------------------------------------- |
  | `AttnConvertor`, `ABIConvertor` | `<UKN>`, `<BOS/EOS>`, `<PAD>`, characters |
  | `CTCConvertor`                  | `<BLK>`, `<UKN>`, characters              |

在 1.0 中，我们不再以任务为边界设计不同的字典和字符序，取而代之的是统一了字符序的 Dictionary，其字符序为 characters, \<BOS/EOS>, \<PAD>, \<UKN>。`CTCConvertor` 中 \<BLK> 被等价替换为 \<PAD>。

- `label_convertor` 中原本支持三种方式初始化字典：`dict_type`、`dict_file` 和 `dict_list`，现在在 `Dictionary` 中被简化为 `dict_file` 一种。同时，我们也把原本在 `dict_type` 中支持的字典格式转化为现在 `dicts/` 目录下的预设字典文件。对应映射如下：

  | MMOCR 0.x: `dict_type` | MMOCR 1.0: 字典路径                    |
  | ---------------------- | -------------------------------------- |
  | DICT90                 | dicts/english_digits_symbols.txt       |
  | DICT91                 | dicts/english_digits_symbols_space.txt |
  | DICT36                 | dicts/lower_english_digits.txt         |
  | DICT37                 | dicts/lower_english_digits_space.txt   |

- `label_converter` 中 `str2tensor()` 的实现被转移到 `ModuleLoss.get_targets()` 中。下面的表格列出了旧版与新版方法实现的对应关系。注意，新旧版的实现并非完全一致。

  | MMOCR 0.x                                                 | MMOCR 1.0                               | 备注                                         |
  | --------------------------------------------------------- | --------------------------------------- | -------------------------------------------- |
  | `ABIConvertor.str2tensor()`, `AttnConvertor.str2tensor()` | `BaseTextRecogModuleLoss.get_targets()` | 原本两个类中的实现存在的差异在新版本中被统一 |
  | `CTCConvertor.str2tensor()`                               | `CTCModuleLoss.get_targets()`           |                                              |

- `label_converter` 中 `tensor2idx()` 的实现被转移到 `Postprocessor.get_single_prediction()` 中。下面的表格列出了旧版与新版方法实现的对应关系。注意，新旧版的实现并非完全一致。

  | MMOCR 0.x                                                 | MMOCR 1.0                                        |
  | --------------------------------------------------------- | ------------------------------------------------ |
  | `ABIConvertor.tensor2idx()`, `AttnConvertor.tensor2idx()` | `AttentionPostprocessor.get_single_prediction()` |
  | `CTCConvertor.tensor2idx()`                               | `CTCPostProcessor.get_single_prediction()`       |

## 关键信息提取

### 关键改动（太长不看版）

- 由于模型的输入发生了变化，旧版模型的权重已经不再能直接应用于 1.0 中。

### KIEDataset & OpensetKIEDataset

- 读取数据的部分被简化到 `WildReceiptDataset` 中。

- 对节点和边作额外处理的部分被转移到了 `LoadKIEAnnotation` 中。

- 使用字典对文本进行转化的部分被转移到了 `SDMGRHead.convert_text()` 中，使用 `Dictionary` 实现。

- 计算文本框之间关系的部分`compute_relation()` 被转移到 `SDMGRHead.compute_relations()` 中，在模型内进行。

- 评估模型表现的部分被简化为 `F1Metric`。

- `OpensetKIEDataset` 中处理模型边输出的部分被整理到 `SDMGRPostProcessor`中。

### SDMGR

- `show_result()` 被整合到 `KIEVisualizer` 中。

- `forward_test()` 中对输出进行后处理的部分被整理到 `SDMGRPostProcessor`中。
