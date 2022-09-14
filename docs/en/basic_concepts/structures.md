# Data Structures and Elements

During the training/testing process of a model, there is often a large amount of data to be passed between modules, and the data required by different tasks or algorithms is usually different. For example, in MMOCR, the text detection task needs to obtain the bounding box annotations of text instances during training, the recognition task needs text annotations, while the key information extraction task needs text category labels and the relationship between items, etc. This makes the interfaces of different tasks or models may be inconsistent, for example:

```python
# Text Detection
for img, img_metas, gt_bboxes in dataloader:
  loss = detector(img, img_metas, gt_bboxes)

# Text Recognition
for img, img_metas, gt_texts in dataloader:
  loss = recognizer(img, img_metas, gt_labels)

# Key Information Extraction
for img, img_metas, gt_bboxes, gt_texts, gt_labels, gt_relations in dataloader:
  loss = kie(img, img_metas, gt_bboxes, gt_texts, gt_labels, gt_relations)
```

From the above code examples, we can see that without encapsulation, the different data required by different tasks and algorithms lead to inconsistent interfaces between their modules, which seriously affects the extensibility and reusability of the library. Therefore, in order to solve the above problem, we use {external+mmengine:doc}`MMEngine: Abstract Data Element <advanced_tutorials/data_element>` to encapsulate the data required for each task into `data_sample`. The base class has implemented basic add/delete/update/check functions and supports data migration between different devices, as well as dictionary-like and tensor-like operations, which also allows the interfaces of different algorithms to be unified in the following form.

```python
for img, data_sample in dataloader:
  loss = model(img, data_sample)
```

Thanks to the unified data structures, the data flow between each module in the algorithm libraries, such as [`visualizer`](./visualizers.md), [`evaluator`](./evaluation.md), [`dataset`](./datasets.md), is greatly simplified. In MMOCR, we have the following conventions for different data types.

- **xxxData**: Single granularity data annotation or model output. Currently MMEngine has three built-in granularities of {external+mmengine:doc}`data elements <advanced_tutorials/data_element>`, including instance-level data (`InstanceData`), pixel-level data (`PixelData`) and image-level label data (`LabelData`). Among the tasks currently supported by MMOCR, text detection and key information extraction tasks use `InstanceData` to encapsulate the bounding boxes and the corresponding box label, while the text recognition task uses `LabelData` to encapsulate the text content.
- **xxxDataSample**: inherited from {external+mmengine:doc}`MMEngine: data base class <advanced_tutorials/data_element>` `BaseDataElement`, used to hold **all** annotation and prediction information that required by a single task. For example, [`TextDetDataSample`](mmocr.structures.textdet_data_sample.TextDetDataSample) for the text detection, [`TextRecogDataSample`](mmocr.structures.textrecog_data_sample.TextRecogDataSample) for text recognition, and [`KIEDataSample`](mmocr.structures.kie_data_sample.KIEDataSample) for the key information extraction task.

In the following, we will introduce the practical application of data elements **xxxData** and data samples **xxxDataSample** in MMOCR, respectively.

## Data Elements - xxxData

`InstanceData` and `LabelData` are the base data elements defined in `MMEngine` to encapsulate different granularity of annotation data or model output. In MMOCR, we have used `InstanceData` and `LabelData` for encapsulating the data types actually used in OCR-related tasks.

### Text Detection - InstanceData

In the text detection task, the detector concentrate on instance-level text samples, so we use `InstanceData` to encapsulate the data needed for this task. Typically, its required training annotation and prediction output contain rectangular or polygonal bounding boxes, as well as bounding box labels. Since the text detection task has only one positive sample class, "text", in MMOCR we use `0` to number this class by default. The following code example shows how to use the `InstanceData` to encapsulate the data used in the text detection task.

```python
import torch
from mmengine.data import InstanceData

# defining gt_instance for encapsulating the ground truth data
gt_instance = InstanceData()
gt_instance.bbox = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
gt_instance.polygons = torch.Tensor([[[0, 0], [10, 0], [10, 10], [0, 10]],
                                    [[10, 10], [20, 10], [20, 20], [10, 20]]])
gt_instance.label = torch.Tensor([0, 0])

# defining pred_instance for encapsulating the prediction data
pred_instances = InstanceData()
pred_polygons, scores = model(input)
pred_instances.polygons = pred_polygons
pred_instances.scores = scores
```

The conventions for the fields in `InstanceData` in MMOCR are shown in the table below. It is important to note that the length of each field in `InstanceData` must be equal to the number of instances `N` in the sample.

|             |                                    |                                                                                                                                                             |
| ----------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Field       | Type                               | Description                                                                                                                                                 |
| bboxes      | `torch.FloatTensor`                | Bounding boxes `[x1, x2, y1, y2]` with the shape `(N, 4)`.                                                                                                  |
| labels      | `torch.LongTensor`                 | Instance label with the shape `(N, )`. By default, MMOCR uses `0` to represent the "text" class.                                                            |
| polygons    | `list[np.array(dtype=np.float32)]` | Polygonal bounding boxes with the shape `(N, )`.                                                                                                            |
| scores      | `torch.Tensor`                     | Confidence scores of the predictions of bounding boxes. `(N, )`.                                                                                            |
| ignored     | `torch.BoolTensor`                 | Whether to ignore the current sample with the shape `(N, )`.                                                                                                |
| texts       | `list[str]`                        | The text content of each instance with the shape `(N, )`，used for e2e text spotting or KIE task.                                                           |
| text_scores | `torch.FloatTensor`                | Confidence score of the predictions of text contents with the shape `(N, )`，used for e2e text spotting task.                                               |
| edge_labels | `torch.IntTensor`                  | The node adjacency matrix with the shape `(N, N)`. In KIE, the optional values for the state between nodes are `-1` (ignored, not involved in loss calculation)，`0` (disconnected) and `1`(connected). |
| edge_scores | `torch.FloatTensor`                | The prediction confidence of each edge in the KIE task, with the shape `(N, N)`.                                                                            |

### Text Recognition - LabelData

For **text recognition** tasks, both labeled content and predicted content are wrapped using `LabelData`.

```python
import torch
from mmengine.data import LabelData

# defining gt_text for encapsulating the ground truth data
gt_text = LabelData()
gt_text.item = 'MMOCR'

# defining pred_text for encapsulating the prediction data
pred_text = LabelData()
index, score = model(input)
text = dictionary.idx2str(index)
pred_text.score = score
pred_text.item = text
```

The conventions for the `LabelData` fields in MMOCR are shown in the following table.

|                |                    |                                                                                                                                                                          |
| -------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Field          | Type               | Description                                                                                                                                                              |
| item           | `str`              | Text content.                                                                                                                                                            |
| score          | `list[float]`      | Confidence socre of the predicted text.                                                                                                                                  |
| indexes        | `torch.LongTensor` | A sequence of text characters encoded by [dictionary](../basic_concepts/models.md#dictionary) and containing all special characters except `<UNK>`.                      |
| padded_indexes | `torch.LongTensor` | If the length of indexes is less than the maximum sequence length and `pad_idx` exists, this field holds the encoded text sequence padded to the maximum sequence length of `max_seq_len`. |

## DataSample xxxDataSample

By defining a uniform data structure, we can easily encapsulate the annotation data and prediction results in a unified way, making data transfer between different modules of the code base easier. In MMOCR, we have designed three data structures based on the data needed in three tasks: [`TextDetDataSample`](mmocr.structures.textdet_data_sample.TextDetDataSample), [`TextRecogDataSample`](mmocr.structures.textrecog_data_sample.TextRecogDataSample), and [`KIEDataSample`](mmocr.structures.kie_data_sample.KIEDataSample). These data structures all inherit from {external+mmengine:doc}`MMEngine: data base class <advanced_tutorials/data_element>` `BaseDataElement`, which is used to hold all annotation and prediction information required by each task.

### Text Detection - TextDetDataSample

[TextDetDataSample](mmocr.structures.textdet_data_sample.TextDetDataSample) is used to encapsulate the data needed for the text detection task. It contains two main fields `gt_instances` and `pred_instances`, which are used to store the annotation information and prediction results respectively.

|                |                                 |                         |
| -------------- | ------------------------------- | ----------------------- |
| Field          | Type                            | Description             |
| gt_instances   | [`InstanceData`](#instancedata) | Annotation information. |
| pred_instances | [`InstanceData`](#instancedata) | Prediction results.     |

The fields of [`InstanceData`](#instancedata) that will be used are:

|          |                                    |                                                                                                  |
| -------- | ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| Field    | Type                               | Description                                                                                      |
| bboxes   | `torch.FloatTensor`                | Bounding boxes `[x1, x2, y1, y2]` with the shape `(N, 4)`.                                       |
| labels   | `torch.LongTensor`                 | Instance label with the shape `(N, )`. By default, MMOCR uses `0` to represent the "text" class. |
| polygons | `list[np.array(dtype=np.float32)]` | Polygonal bounding boxes with the shape `(N, )`.                                                 |
| scores   | `torch.Tensor`                     | Confidence scores of the predictions of bounding boxes. `(N, )`.                                 |
| ignored  | `torch.BoolTensor`                 | Boolean flags with the shape `(N, )`, indicating whether to ignore the current sample.           |

Since text detection models usually only output one of the bboxes/polygons, we only need to make sure that one of these two is assigned a value.

The following sample code demonstrates the use of `TextDetDataSample`.

```python
import torch
from mmengine.data import TextDetDataSample

data_sample = TextDetDataSample()
# Define the ground truth data
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
gt_instances = InstanceData(metainfo=img_meta)
gt_instances.bboxes = torch.rand((5, 4))
gt_instances.labels = torch.zeros((5,), dtype=torch.long)
data_sample.gt_instances = gt_instances

# Define the prediction data
pred_instances = InstanceData()
pred_instances.bboxes = torch.rand((5, 4))
pred_instances.labels = torch.zeros((5,), dtype=torch.long)
data_sample.pred_instances = pred_instances
```

### Text Recognition - TextRecogDataSample

[`TextRecogDataSample`](mmocr.structures.textrecog_data_sample.TextRecogDataSample) is used to encapsulate the data for the text recognition task. It has two fields, `gt_text` and `pred_text` , which are used to store annotation information and prediction results, respectively.

|           |                                            |                     |
| --------- | ------------------------------------------ | ------------------- |
| Field     | Type                                       | Description         |
| gt_text   | [`LabelData`](#text-recognition-labeldata) | Label information.  |
| pred_text | [`LabelData`](#text-recognition-labeldata) | Prediction results. |

The following sample code demonstrates the use of [`TextRecogDataSample`](mmocr.structures.textrecog_data_sample.TextRecogDataSample).

```python
import torch
from mmengine.data import TextRecogDataSample

data_sample = TextRecogDataSample()
# Define the ground truth data
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
gt_text = LabelData(metainfo=img_meta)
gt_text.item = 'mmocr'
data_sample.gt_text = gt_text

# Define the prediction data
pred_text = LabelData(metainfo=img_meta)
pred_text.item = 'mmocr'
data_sample.pred_text = pred_text
```

The fields of `LabelData` that will be used are:

|                |                     |                                                                                                                                                                         |
| -------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Field          | Type                | Description                                                                                                                                                             |
| item           | `list[str]`         | The text corresponding to the instance, of length (N, ), for end-to-end OCR tasks and KIE                                                                               |
| score          | `torch.FloatTensor` | Confidence of the text prediction, of length (N, ), for the end-to-end OCR task                                                                                         |
| indexes        | `torch.LongTensor`  | A sequence of text characters encoded by [dictionary](../basic_concepts/models.md#dictionary) and containing all special characters except `<UNK>`.                     |
| padded_indexes | `torch.LongTensor`  | If the length of indexes is less than the maximum sequence length and `pad_idx` exists, this field holds the encoded text sequence padded to the maximum sequence length of `max_seq_len`. |

### Key Information Extraction - KIEDataSample

[`KIEDataSample`](mmocr.structures.kie_data_sample.KIEDataSample) is used to encapsulate the data needed for the KIE task. It also contains two fields, `gt_instances` and `pred_instances`, which are used to store annotation information and prediction results respectively.

|                |                                                |                         |
| -------------- | ---------------------------------------------- | ----------------------- |
| Field          | Type                                           | Description             |
| gt_instances   | [`InstanceData`](#text-detection-instancedata) | Annotation information. |
| pred_instances | [`InstanceData`](#text-detection-instancedata) | Prediction results.     |

The [`InstanceData`](#text-detection-instancedata) fields that will be used by this task are shown in the following table.

|             |                     |                                                                                                                                                                            |
| ----------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Field       | Type                | Description                                                                                                                                                                |
| bboxes      | `torch.FloatTensor` | Bounding boxes `[x1, x2, y1, y2]` with the shape `(N, 4)`.                                                                                                                 |
| labels      | `torch.LongTensor`  | Instance label with the shape `(N, )`.                                                                                                                                     |
| texts       | `list[str]`         | The text content of each instance with the shape `(N, )`，used for e2e text spotting or KIE task.                                                                          |
| edge_labels | `torch.IntTensor`   | The node adjacency matrix with the shape `(N, N)`. In the KIE task, the optional values for the state between nodes are `-1` (ignored, not involved in loss calculation)，`0` (disconnected) and `1`(connected). |
| edge_scores | `torch.FloatTensor` | The prediction confidence of each edge in the KIE task, with the shape `(N, N)`.                                                                                           |
| scores      | `torch.FloatTensor` | The confidence scores for node label predictions, with the shape `(N,)`.                                                                                                   |

```{warning}
Since there is no unified standard for model implementation of KIE tasks, the design currently considers only [SDMGR](../../../configs/kie/sdmgr/README.md) model usage scenarios. Therefore, the design is subject to change as we support more KIE models.
```

The following sample code shows the use of [`KIEDataSample`](mmocr.structures.kie_data_sample.KIEDataSample).

```python
import torch
from mmengine.data import KIEDataSample

data_sample = KIEDataSample()
# Define the ground truth data
img_meta = dict(img_shape=(800, 1196, 3),pad_shape=(800, 1216, 3))
gt_instances = InstanceData(metainfo=img_meta)
gt_instances.bboxes = torch.rand((5, 4))
gt_instances.labels = torch.zeros((5,), dtype=torch.long)
gt_instances.texts = ['text1', 'text2', 'text3', 'text4', 'text5']
gt_instances.edge_lebels = torch.randint(-1, 2, (5, 5))
data_sample.gt_instances = gt_instances

# Define the prediction data
pred_instances = InstanceData()
pred_instances.bboxes = torch.rand((5, 4))
pred_instances.labels = torch.rand((5,))
pred_instances.edge_labels = torch.randint(-1, 2, (10, 10))
pred_instances.edge_scores = torch.rand((10, 10))
data_sample.pred_instances = pred_instances
```
