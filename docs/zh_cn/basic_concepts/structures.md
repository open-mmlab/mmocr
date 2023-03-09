# 数据元素与数据结构

MMOCR 基于 {external+mmengine:doc}`MMEngine: 抽象数据接口 <advanced_tutorials/data_element>` 将各任务所需的数据统一封装入 `data_sample` 中。MMEngine 的抽象数据接口实现了基础的增/删/改/查功能，且支持不同设备间的数据迁移，也支持了类字典和张量的操作，充分满足了数据的日常使用需求，这也使得不同算法的数据接口可以得到统一。

得益于统一的数据封装，算法库内的 [`visualizer`](./visualizers.md)，[`evaluator`](./evaluation.md)，[`dataset`](./datasets.md) 等各个模块间的数据流通都得到了极大的简化。在 MMOCR 中，我们对数据接口类型作出以下约定：

- **xxxData**: 单一粒度的数据标注或模型输出。目前 MMEngine 内置了三种粒度的{external+mmengine:doc}`数据元素 <advanced_tutorials/data_element>`，包括实例级数据（`InstanceData`），像素级数据（`PixelData`）以及图像级的标签数据（`LabelData`）。在 MMOCR 目前支持的任务中，文本检测以及关键信息抽取任务使用 `InstanceData` 来封装文本实例的检测框及对应标签，而文本识别任务则使用了 `LabelData` 来封装文本内容。
- **xxxDataSample**: 继承自 {external+mmengine:doc}`MMEngine: 数据基类 <advanced_tutorials/data_element>` `BaseDataElement`，用于保存单个任务的训练或测试样本的**所有**标注及预测信息。如文本检测任务的数据样本类 [`TextDetDataSample`](mmocr.structures.textdet_data_sample.TextDetDataSample)，文本识别任务的数据样本类 [`TextRecogDataSample`](mmocr.structures.textrecog_data_sample.TextRecogDataSample)，以及关键信息抽任务的数据样本类 [`KIEDataSample`](mmocr.structures.kie_data_sample.KIEDataSample)。

下面，我们将分别介绍数据元素 **xxxData** 与数据样本 **xxxDataSample** 在 MMOCR 中的实际应用。

## 数据元素 xxxData

`InstanceData` 和 `LabelData` 是 `MMEngine`中定义的基础数据元素，用于封装不同粒度的标注数据或模型输出。在 MMOCR 中，我们针对不同任务中实际使用的数据类型，分别采用了 `InstanceData` 与 `LabelData` 进行了封装。

### InstanceData

在**文本检测**任务中，检测器关注的是实例级别的文字样本，因此我们使用 `InstanceData` 来封装该任务所需的数据。其所需的训练标注和预测输出通常包含了矩形或多边形边界盒，以及边界盒标签。由于文本检测任务只有一种正样本类，即 “text”，在 MMOCR 中我们默认使用 `0` 来编号该类别。以下代码示例展示了如何使用 `InstanceData` 数据抽象接口来封装文本检测任务中使用的数据类型。

```python
import torch
from mmengine.structures import InstanceData

# 定义 gt_instance 用于封装边界盒的标注信息
gt_instance = InstanceData()
gt_instance.bbox = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
gt_instance.polygons = torch.Tensor([[[0, 0], [10, 0], [10, 10], [0, 10]],
                                    [[10, 10], [20, 10], [20, 20], [10, 20]]])
gt_instance.label = torch.Tensor([0, 0])

# 定义 pred_instance 用于封装模型的输出信息
pred_instances = InstanceData()
pred_polygons, scores = model(input)
pred_instances.polygons = pred_polygons
pred_instances.scores = scores
```

MMOCR 中对 `InstanceData` 字段的约定如下表所示。值得注意的是，`InstanceData` 中的各字段的长度必须为与样本中的实例个数 `N` 相等。

|             |                                    |                                                                                                                                      |
| ----------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 字段        | 类型                               | 说明                                                                                                                                 |
| bboxes      | `torch.FloatTensor`                | 文本边界框 `[x1, y1, x2, y2]`，形状为 `(N, 4)`。                                                                                     |
| labels      | `torch.LongTensor`                 | 实例的类别，长度为 `(N, )`。MMOCR 中默认使用 `0` 来表示正样本类，即 “text” 类。                                                      |
| polygons    | `list[np.array(dtype=np.float32)]` | 表示文本实例的多边形，列表长度为 `(N, )`。                                                                                           |
| scores      | `torch.Tensor`                     | 文本实例检测框的置信度，长度为 `(N, )`。                                                                                             |
| ignored     | `torch.BoolTensor`                 | 是否在训练中忽略当前文本实例，长度为 `(N, )`。                                                                                       |
| texts       | `list[str]`                        | 实例对应的文本，长度为 `(N, )`，用于端到端 OCR 任务和 KIE。                                                                          |
| text_scores | `torch.FloatTensor`                | 文本预测的置信度，长度为`(N, )`，用于端到端 OCR 任务。                                                                               |
| edge_labels | `torch.IntTensor`                  | 节点的邻接矩阵，形状为 `(N, N)`。在 KIE 任务中，节点之间状态的可选值为 `-1` （忽略，不参与 loss 计算），`0` （断开）和 `1`（连接）。 |
| edge_scores | `torch.FloatTensor`                | 用于 KIE 任务中每条边的预测置信度，形状为 `(N, N)`。                                                                                 |

### LabelData

对于**文字识别**任务，标注内容和预测内容都会使用 `LabelData` 进行封装。

```python
import torch
from mmengine.data import LabelData

# 定义一个 gt_text 用于封装标签文本内容
gt_text = LabelData()
gt_text.item = 'MMOCR'

# 定义一个 pred_text 对象用于封装预测文本以及置信度
pred_text = LabelData()
index, score = model(input)
text = dictionary.idx2str(index)
pred_text.score = score
pred_text.item = text
```

MMOCR 中对 `LabelData` 字段的约定如下表所示：

|                |                    |                                                                                                                            |
| -------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| 字段           | 类型               | 说明                                                                                                                       |
| item           | `str`              | 文本内容。                                                                                                                 |
| score          | `list[float]`      | 预测的文本内容的置信度。                                                                                                   |
| indexes        | `torch.LongTensor` | 文本字符经过[字典](../basic_concepts/models.md#dictionary)编码后的序列，且包含了除 `<UNK>` 以外的所有特殊字符。            |
| padded_indexes | `torch.LongTensor` | 如果 indexes 的长度小于最大序列长度，且 `pad_idx` 存在时，该字段保存了填充至最大序列长度 `max_seq_len`的编码后的文本序列。 |

## 数据样本 xxxDataSample

通过定义统一的数据结构，我们可以方便地将标注数据和预测结果进行统一封装，使代码库不同模块间的数据传递更加便捷。在 MMOCR 中，我们基于现在支持的三个任务及其所需要的数据分别封装了三种数据抽象，包括文本检测任务数据抽象 [`TextDetDataSample`](mmocr.structures.textdet_data_sample.TextDetDataSample)，文本识别任务数据抽象 [`TextRecogDataSample`](mmocr.structures.textrecog_data_sample.TextRecogDataSample)，以及关键信息抽取任务数据抽象 [`KIEDataSample`](mmocr.structures.kie_data_sample.KIEDataSample)。这些数据抽象均继承自 {external+mmengine:doc}`MMEngine: 数据基类 <advanced_tutorials/data_element>` `BaseDataElement`，用于保存单个任务的训练或测试样本的所有标注及预测信息。

### 文本检测任务数据抽象 TextDetDataSample

[TextDetDataSample](mmocr.structures.textdet_data_sample.TextDetDataSample) 用于封装文字检测任务所需的数据，其主要包含了两个字段 `gt_instances` 与 `pred_instances`，分别用于存放标注信息与预测结果。

|                |                                 |            |
| -------------- | ------------------------------- | ---------- |
| 字段           | 类型                            | 说明       |
| gt_instances   | [`InstanceData`](#instancedata) | 标注信息。 |
| pred_instances | [`InstanceData`](#instancedata) | 预测结果。 |

其中会用到的 [`InstanceData`](#instancedata) 约定字段有：

|          |                                    |                                                                                  |
| -------- | ---------------------------------- | -------------------------------------------------------------------------------- |
| 字段     | 类型                               | 说明                                                                             |
| bboxes   | `torch.FloatTensor`                | 文本边界框 `[x1, y1, x2, y2]`，形状为 `(N, 4)`。                                 |
| labels   | `torch.LongTensor`                 | 实例的类别，长度为 `(N, )`。在 MMOCR 中通常使用 `0` 来表示正样本类，即 “text” 类 |
| polygons | `list[np.array(dtype=np.float32)]` | 表示文本实例的多边形，列表长度为 `(N, )`。                                       |
| scores   | `torch.Tensor`                     | 文本实例任务预测的检测框的置信度，长度为 `(N, )`。                               |
| ignored  | `torch.BoolTensor`                 | 是否在训练中忽略当前文本实例，长度为 `(N, )`。                                   |

由于文本检测模型通常只会输出 bboxes/polygons 中的一项，因此我们只需确保这两项中的一个被赋值即可。

以下示例代码展示了 `TextDetDataSample` 的使用方法：

```python
import torch
from mmengine.data import TextDetDataSample

data_sample = TextDetDataSample()
# 指定当前图片的标注信息
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
gt_instances = InstanceData(metainfo=img_meta)
gt_instances.bboxes = torch.rand((5, 4))
gt_instances.labels = torch.zeros((5,), dtype=torch.long)
data_sample.gt_instances = gt_instances

# 指定当前图片的预测信息
pred_instances = InstanceData()
pred_instances.bboxes = torch.rand((5, 4))
pred_instances.labels = torch.zeros((5,), dtype=torch.long)
data_sample.pred_instances = pred_instances
```

### 文本识别任务数据抽象 TextRecogDataSample

[`TextRecogDataSample`](mmocr.structures.textrecog_data_sample.TextRecogDataSample) 用于封装文字识别任务的数据。它有两个属性，`gt_text` 和 `pred_text` , 分别用于存放标注信息和预测结果。

|           |                           |            |
| --------- | ------------------------- | ---------- |
| 字段      | 类型                      | 说明       |
| gt_text   | [`LabelData`](#labeldata) | 标注信息。 |
| pred_text | [`LabelData`](#labeldata) | 预测结果。 |

以下示例代码展示了 [`TextRecogDataSample`](mmocr.structures.textrecog_data_sample.TextRecogDataSample) 的使用方法：

```python
import torch
from mmengine.data import TextRecogDataSample

data_sample = TextRecogDataSample()
# 指定当前图片的标注信息
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
gt_text = LabelData(metainfo=img_meta)
gt_text.item = 'mmocr'
data_sample.gt_text = gt_text

# 指定当前图片的预测结果
pred_text = LabelData(metainfo=img_meta)
pred_text.item = 'mmocr'
data_sample.pred_text = pred_text
```

其中会用到的 `LabelData` 字段有：

|                |                     |                                                                                                                            |
| -------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 字段           | 类型                | 说明                                                                                                                       |
| item           | `list[str]`         | 实例对应的文本，长度为 (N, ) ，用于端到端 OCR 任务和 KIE                                                                   |
| score          | `torch.FloatTensor` | 文本预测的置信度，长度为 (N, )，用于端到端 OCR 任务                                                                        |
| indexes        | `torch.LongTensor`  | 文本字符经过[字典](../basic_concepts/models.md#dictionary)编码后的序列，且包含了除 `<UNK>` 以外的所有特殊字符。            |
| padded_indexes | `torch.LongTensor`  | 如果 indexes 的长度小于最大序列长度，且 `pad_idx` 存在时，该字段保存了填充至最大序列长度 `max_seq_len`的编码后的文本序列。 |

### 关键信息抽取任务数据抽象 KIEDataSample

[`KIEDataSample`](mmocr.structures.kie_data_sample.KIEDataSample) 用于封装 KIE 任务所需的数据，其同样约定了两个属性，即 `gt_instances` 与 `pred_instances`，分别用于存放标注信息与预测结果。

|                |                                 |            |
| -------------- | ------------------------------- | ---------- |
| 字段           | 类型                            | 说明       |
| gt_instances   | [`InstanceData`](#instancedata) | 标注信息。 |
| pred_instances | [`InstanceData`](#instancedata) | 预测结果。 |

该任务会用到的 [`InstanceData`](#instancedata) 字段如下表所示：

|             |                     |                                                                                                                                               |
| ----------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 字段        | 类型                | 说明                                                                                                                                          |
| bboxes      | `torch.Tensor`      | 文本边界框 `[x1, y1, x2, y2]`，形状为 `(N, 4)`。                                                                                              |
| labels      | `torch.LongTensor`  | 实例的类别，长度为 `(N, )`。在 MMOCR 中通常为 0，即 “text” 类。                                                                               |
| texts       | `list[str]`         | 实例对应的文本，长度为 `(N, )` ，用于端到端 OCR 任务和 KIE 任务。                                                                             |
| edge_labels | `torch.IntTensor`   | 节点之间的邻接矩阵，形状为 `(N, N)`。在 KIE 任务中，节点之间状态的可选值为 `-1` （不关心，且不参与 loss 计算），`0` （断开）和 `1` （连接）。 |
| edge_scores | `torch.FloatTensor` | 每条边的预测置信度，形状为 `(N, N)`。                                                                                                         |
| scores      | `torch.FloatTensor` | 节点标签的预测置信度, 形状为 `(N,)`。                                                                                                         |

```{warning}
由于 KIE 任务的模型实现尚未有统一标准，该设计目前仅考虑了 [SDMGR](../../../configs/kie/sdmgr/README.md) 模型的使用场景。因此，该设计有可能在我们支持更多 KIE 模型后产生变动。
```

以下示例代码展示了 [`KIEDataSample`](mmocr.structures.kie_data_sample.KIEDataSample) 的使用方法。

```python
import torch
from mmengine.data import KIEDataSample

data_sample = KIEDataSample()
# 指定当前图片的标注信息
img_meta = dict(img_shape=(800, 1196, 3),pad_shape=(800, 1216, 3))
gt_instances = InstanceData(metainfo=img_meta)
gt_instances.bboxes = torch.rand((5, 4))
gt_instances.labels = torch.zeros((5,), dtype=torch.long)
gt_instances.texts = ['text1', 'text2', 'text3', 'text4', 'text5']
gt_instances.edge_lebels = torch.randint(-1, 2, (5, 5))
data_sample.gt_instances = gt_instances

# 指定当前图片的预测信息
pred_instances = InstanceData()
pred_instances.bboxes = torch.rand((5, 4))
pred_instances.labels = torch.rand((5,))
pred_instances.edge_labels = torch.randint(-1, 2, (10, 10))
pred_instances.edge_scores = torch.rand((10, 10))
data_sample.pred_instances = pred_instances
```
