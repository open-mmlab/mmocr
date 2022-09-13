# Data Structures and Elements

During the training/testing process of a model, there is often a large amount of data to be passed between modules, and the data required by different tasks or algorithms is usually different. For example, in MMOCR, the text detection task needs to obtain the bounding box annotations of text instances during training, the recognition task needs text content annotations, while the key information extraction task needs text category labels and the relationship between items, etc. This makes the interfaces of different tasks or models may be inconsistent, for example:

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

From the above code examples, we can see that without encapsulation, the different data required by different tasks and algorithms lead to inconsistent interfaces between their modules, which seriously affects the extensibility and reusability of the library. Therefore, in order to solve the above problem, we use {external+mmengine:doc}`MMEngine: Abstract Data Element <advanced_tutorials/data_element>` to encapsulate the data required for each task into `data_sample`. The base class has implemented basic add/delete/update/check functions and supports data migration between different devices, as well as dictionary-like and tensor-like operations, fully satisfying the daily use requirements of data, which also allows the interfaces of different algorithms to be unified in the following form.

```python
for img, data_sample in dataloader:
  loss = model(img, data_sample)
```

Thanks to the unified data structures, the data flow between each module in the algorithm libraries, such as [`visualizer`](./visualizers.md), [`evaluator`](./evaluation.md), [`dataset`](./datasets.md), is greatly simplified. In MMOCR, we make the following conventions for data interface types.

- **xxxData**: Single granularity data annotation or model output. Currently MMEngine has three built-in granularities of {external+mmengine:doc}`data elements <advanced_tutorials/data_element>`, including instance-level data (`InstanceData`), pixel-level data (`PixelData`) and image-level label data (`LabelData`). Among the tasks currently supported by MMOCR, the text detection task uses `InstanceData` to encapsulate the bounding boxes and the corresponding box label, while the text recognition task uses `LabelData` to encapsulate the text content.
- **xxxDataSample**: inherited from {external+mmengine:doc}`MMEngine: data base class <advanced_tutorials/data_element>` `BaseDataElement`, used to hold **all** annotation and prediction information that required by a single task. For example, [`TextDetDataSample`](mmocr.structures.TextDetDataSample) for the text detection, \[`TextRecogDataSample`\](mmocr.structures. TextRecogDataSample) for text recognition, and [`KIEDataSample`](mmocr.structures.KIEDataSample) for the key information extraction task.

In the following, we will introduce the practical application of data elements **xxxData** and data samples **xxxDataSample** in MMOCR, respectively.

## Data elements xxxData

`InstanceData` and `LabelData` are the base data elements defined in `MMEngine` to encapsulate different granularity of annotation data or model output. In MMOCR, we have used `InstanceData` and `LabelData` for encapsulating the data types actually used in OCR-related tasks.

### Text Detection InstanceData

In the text detection task, the detector concentrate on instance-level text samples, so we use `InstanceData` to encapsulate the data needed for this task. Typically, its required training annotation and prediction output contains rectangular or polygonal bounding boxes, as well as bounding box labels. Since the text detection task has only one positive sample class, "text", in MMOCR we use `0` to number this class by default. The following code example shows how to use the `InstanceData` data abstraction interface to encapsulate the data types used in the text detection task.

```python
import torch
from mmengine.data import InstanceData

# defining gt_instance for encapsulate the ground truth data
gt_instance = InstanceData()
gt_instance.bbox = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
gt_instance.polygons = torch.Tensor([[[0, 0], [10, 0], [10, 10], [0, 10]],
                                    [[10, 10], [20, 10], [20, 20], [10, 20]]])
gt_instance.label = torch.Tensor([0, 0])

# defining pred_instance for encapsulate the prediction data
pred_instances = InstanceData()
pred_polygons, scores = model(input)
pred_instances.polygons = pred_polygons
pred_instances.scores = scores
```

The conventions for the `InstanceData` fields in MMOCR are shown in the table below. It is important to note that the length of each field in `InstanceData` must be equal to the number of instances in the sample `N`.

|             |                                    |                                                                                                                                                             |
| ----------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Field       | Type                               | Description                                                                                                                                                 |
| bboxes      | `torch.Tensor(float32)`            | Bounding boxes `[x1, x2, y1, y2]` with the shape `(N, 4)`.                                                                                                  |
| labels      | `torch.LongTensor`                 | Instance label with the shape `(N, )`. By default, MMOCR uses `0` to represent the "text" class.                                                            |
| polygons    | `list[np.array(dtype=np.float32)]` | Polygonal bounding boxes with the shape `(N, )`.                                                                                                            |
| scores      | `torch.Tensor`                     | Confidence scores of the predictions of bounding boxes. `(N, )`.                                                                                            |
| ignored     | `torch.BoolTensor`                 | Whether to ignore the current sample with the shape `(N, )`.                                                                                                |
| texts       | `list[str]`                        | The textual content of each instance with the shape `(N, )`，used for e2e text spotting or KIE task.                                                        |
| text_scores | `torch.FloatTensor`                | Confidence score of the predictions of text contents with the shape `(N, )`，used for e2e text spotting task.                                               |
| edge_labels | `torch.IntTensor`                  | The adjacency matrix between nodes with the shape `(N, N)`. In the KIE task, the optional values for the state between nodes are `-1` (ignored, not involved in loss calculation)，`0` (disconnected) and `1`(connected). |
| edge_scores | `torch.FloatTensor`                | The prediction confidence of each edge in the KIE task, with the shape `(N, N)`.                                                                            |

### Text Recognition LabelData

For **text recognition** tasks, both labeled content and predicted content are wrapped using `LabelData`.

```python
import torch
from mmengine.data import LabelData

# defining gt_text for encapsulate the ground truth data
gt_text = LabelData()
gt_text.item = 'MMOCR'

# defining pred_text for encapsulate the prediction data
pred_text = LabelData()
index, score = model(input)
text = dictionary.idx2str(index)
pred_text.score = score
pred_text.item = text
```

The conventions for the `LabelData` field in MMOCR are shown in the following table.

|                |                    |                                                                                                                                                                          |
| -------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Field          | Type               | Description                                                                                                                                                              |
| item           | `str`              | Text content.                                                                                                                                                            |
| score          | `list[float]`      | Confidence socre of the predicted text.                                                                                                                                  |
| indexes        | `torch.LongTensor` | A sequence of text characters encoded by [dictionary](#TODO) and containing all special characters except `<UNK>`.                                                       |
| padded_indexes | `torch.LongTensor` | If the length of indexes is less than the maximum sequence length and `pad_idx` exists, this field holds the encoded text sequence padded to the maximum sequence length of `max_seq_len`. |

## DataSample xxxDataSample

By defining a uniform data structure, we can easily encapsulate the annotation data and prediction results in a uniform way, making data transfer between different modules of the code base easier. In MMOCR, we have encapsulated three data abstractions based on the three tasks we now support and the data they need, including the text detection data abstraction [`TextDetDataSample`](mmocr.structures.TextDetDataSample), the text recognition data abstraction [`TextRecogDataSample`](mmocr.structures.TextRecogDataSample), and the key information extraction data abstraction [`KIEDataSample`](mmocr.structures.KIEDataSample). These data abstractions all inherit from {external+mmengine:doc}`MMEngine: data base class <advanced_tutorials/data_element>` `BaseDataElement`, which is used to hold all annotation and prediction information required by each task.

### Text Detection Data Abstraction TextDetDataSample

[TextDetDataSample](mmocr.structures.TextDetDataSample) is used to encapsulate the data needed for the text detection task. It contains two main fields `gt_instances` and `pred_instances`, which are used to store the prediction results and annotation information respectively.

|                |                                 |                         |
| -------------- | ------------------------------- | ----------------------- |
| Field          | Type                            | Description             |
| gt_instances   | [`InstanceData`](#instancedata) | Annotation information. |
| pred_instances | [`InstanceData`](#instancedata) | The predicted result.   |

The fields of [`InstanceData`](#instancedata) that will be used are:

|          |                                    |                                                                                                  |
| -------- | ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| Field    | Type                               | Description                                                                                      |
| bboxes   | `torch.Tensor(float32)`            | Bounding boxes `[x1, x2, y1, y2]` with the shape `(N, 4)`.                                       |
| labels   | `torch.LongTensor`                 | Instance label with the shape `(N, )`. By default, MMOCR uses `0` to represent the "text" class. |
| polygons | `list[np.array(dtype=np.float32)]` | Polygonal bounding boxes with the shape `(N, )`.                                                 |
| scores   | `torch.Tensor`                     | Confidence scores of the predictions of bounding boxes. `(N, )`.                                 |
| ignored  | `torch.BoolTensor`                 | Whether to ignore the current sample with the shape `(N, )`.                                     |

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

[`TextDetDataSample`](mmocr.structures.TextRecogDataSample) is the basic data structure used for model inference and training in text detection tasks. The following code shows the flow of [`TextDetDataSample`](mmocr.structures.TextRecogDataSample) between the modules of the model, using DBNet as an example.

```python
# Step 1: DBHead, input the initial image img and data_samples from the data pipeline, and return the output of the network part of the model
def forward(self, img: Tensor,
        data_samples: Optional[List[TextDetDataSample]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:

# (Training) Step 2: DBModuleLoss, input the output of the DBNet network part and data_samples
def forward(self, preds: Tuple[Tensor, Tensor, Tensor],
                data_samples: Sequence[TextDetDataSample]) -> Dict:

# (Test) Step 2: DBPostProcessor
def get_text_instances(self, pred_results: Tuple[Tensor, Tensor, Tensor],
                           data_sample: TextDetDataSample
                           ) -> TextDetDataSample:
```

As you can see from the above example, [`TextDetDataSample`](mmocr.structures.TextRecogDataSample) is used throughout the training and testing process of the detection model, and it encapsulates the data needed for the whole process of the text detection task.

### Text Recognition Task Data Abstraction TextRecogDataSample

[`TextRecogDataSample`](mmocr.structures.TextRecogDataSample) is used to encapsulate the data for the text recognition task. It has two properties, `gt_text` and `pred_text` , which are used to store prediction results and annotation information respectively.

|           |                           |                        |
| --------- | ------------------------- | ---------------------- |
| Field     | Type                      | Description            |
| gt_text   | [`LabelData`](#labeldata) | Label information.     |
| pred_text | [`LabelData`](#labeldata) | The prediction result. |

The following sample code demonstrates the use of [`TextRecogDataSample`](mmocr.structures.TextRecogDataSample).

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

The fields of `Labelata` that will be used are:

|       |                     |                                                                                           |
| ----- | ------------------- | ----------------------------------------------------------------------------------------- |
| Field | Type                | Description                                                                               |
| item  | `list[str]`         | The text corresponding to the instance, of length (N, ), for end-to-end OCR tasks and KIE |
| score | `torch.FloatTensor` | Confidence of the text prediction, of length (N, ), for the end-to-end OCR task           |

Similarly, [`TextRecogDataSample`](mmocr.structures.TextRecogDataSample) runs through the entire training and testing process of the recognition model, as follows.

```python
# Encoder
def forward(self, feature: torch.Tensor,
                data_samples: List[TextRecogDataSample]) -> torch.Tensor:

# Decoder
def forward_train(self,
                      feat: Optional[torch.Tensor] = None,
                      out_enc: torch.Tensor = None,
                      data_samples: Sequence[TextRecogDataSample] = None
                      ) -> torch.Tensor:

# Module Loss
def forward(self, outputs: torch.Tensor,
                data_samples: Sequence[TextRecogDataSample]) -> Dict:

# Post Processor
def get_single_prediction(
        self,
        probs: torch.Tensor,
        data_sample: Optional[TextRecogDataSample] = None,
    ) -> Tuple[Sequence[int], Sequence[float]]:
```

### Key Information Extraction Task Data Abstraction KIEDataSample

[`KIEDataSample`](mmocr.structures.KIEDataSample) is used to encapsulate the data needed for the KIE task. It also agrees on two properties, `gt_instances` and `pred_instances`, which are used to store annotation information and prediction results respectively.

|                |                                 |             |
| -------------- | ------------------------------- | ----------- |
| Field          | Type                            | Description |
| gt_instances   | [`InstanceData`](#instancedata) | Annotation  |
| pred_instances | [`InstanceData`](#instancedata) | Prediction  |

The [`InstanceData`](#instancedata) fields that will be used by this task are shown in the following table.

|             |                         |                                                                                                                                                                        |
| ----------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Field       | Type                    | Description                                                                                                                                                            |
| bboxes      | `torch.Tensor(float32)` | Bounding boxes `[x1, x2, y1, y2]` with the shape `(N, 4)`.                                                                                                             |
| labels      | `torch.LongTensor`      | Instance label with the shape `(N, )`. By default, MMOCR uses `0` to represent the "text" class.                                                                       |
| texts       | `list[str]`             | The textual content of each instance with the shape `(N, )`，used for e2e text spotting or KIE task.                                                                   |
| edge_labels | `torch.IntTensor`       | The adjacency matrix between nodes with the shape `(N, N)`. In the KIE task, the optional values for the state between nodes are `-1` (ignored, not involved in loss calculation)，`0` (disconnected) and `1`(connected). |
| edge_scores | `torch.FloatTensor`     | The prediction confidence of each edge in the KIE task, with the shape `(N, N)`.                                                                                       |

Since there is no unified standard for model implementation of KIE tasks, the design currently considers only [SDMGR](../../../configs/kie/sdmgr/README.md) model usage scenarios. Therefore, the design is subject to change as we support more KIE models.

The following sample code shows the use of [`KIEDataSample`](mmocr.structures.KIEDataSample).

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
