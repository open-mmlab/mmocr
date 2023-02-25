# Inferencers

In OpenMMLab, all the inference operations are unified into a new inference - `Inferencer`. `Inferencer` is designed to expose a neat and simple API to users, and shares very similar interface across different OpenMMLab libraries.

In MMOCR, Inferencers are constructed in different levels of task abstraction.

- Task-specific Inferencer: Following OpenMMLab's convention, each fundamental task in MMOCR has its Inferencer, namely `TextDetInferencer`, `TextRecInferencer`, `TextSpottingInferencer`, and `KIEInferencer`. They are designed to perform inference on a single task, and can be chained together to perform inference on a series of tasks. They also share very similar interface, have standard input/output protocol, and overall follow the OpenMMLab design.
- [`MMOCRInferencer`](../user_guides/inference.md): We also provide `MMOCRInferencer`, a convenient inference interface only designed for MMOCR. It encapsulates and chains all the Inferencers in MMOCR, so users can use this Inferencer to perform a series of tasks on an image and directly get the final result in an end-to-end manner. *However, it has a relatively different interface from other task-specific Inferencers, and some of standard Inferencer functionalities might be sacrificed for the sake of simplicity.*

For new users, we recommend using [`MMOCRInferencer`](../user_guides/inference.md) to test out different combinations of models.

If you are a developer and wish to integrate the models into your own project, we recommend using task-specific Inferencers, as they are more flexible and standardized, equipped with full functionalities.

This page will introduce the usage of task-specific Inferencers.

## Basic Usage

In general, all the task-specific Inferencers across OpenMMLab share a very similar interface. The following example shows how to use `TextDetInferencer` to perform inference on a single image.

```python
>>> from mmocr.apis import TextDetInferencer
>>> # Load models into memory
>>> ocr = TextDetInferencer(model='DBNet')
>>> # Inference
>>> ocr('demo/demo_text_ocr.jpg', show=True)
```

## Model Initialization

Every `Inferencer` reserves two arguments, `model` and `weights`, for initialization. and there are many ways to initialize a model for inference.

- `model` takes either the name of a model, or the path to a config file as input. The name of a model is obtained from the model's metafile ([Example](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/metafile.yml)) indexed from [model-index.yml](https://github.com/open-mmlab/mmocr/blob/1.x/model-index.yml). You can find the list of available name choices [here](../modelzoo.md#weights).

  ```{note}
  For convenience, we abbreviate the names of some commonly-used models in the "Alias" field of its metafile, which Inferencer can use to index a model as well.
  ```

- `weights` accepts the path to a weight file.

There are many ways to initialize a model.

- To infer with MMOCR's pre-trained model,  you can pass its name to `model`. The weights will be automatically downloaded and loaded from OpenMMLab's model zoo. Check [Weights](../modelzoo.md#weights) for available model names.

  ```python
  >>> from mmocr.apis import TextDetInferencer
  >>> inferencer = TextDetInferencer(model='DBNet')
  ```

  ```{note}
  The model type must match the Inferencer type.
  ```

  To load the custom weight, you can also pass its path/url to `weights`.

  ```python
  >>> inferencer = TextDetInferencer(model='DBNet', weights='path/to/dbnet.pth')
  ```

- To load custom config and weight, you can pass the path to the config file to `model` and the path to the weight to `weights`.

  ```python
  >>> inferencer = TextDetInferencer(model='path/to/dbnet_config.py', weights='path/to/dbnet.pth')
  ```

- By default, [MMEngine](https://github.com/open-mmlab/mmengine/) dumps config to the weight. If you have a weight trained on MMEngine, you can also pass the path to the weight file to `weights` without specifying `model`:

  ```python
  >>> # It will raise an error if the config file cannot be found in the weight
  >>> inferencer = TextDetInferencer(weights='path/to/dbnet.pth')
  ```

- Passing config file to `xxx` without specifying the weight path `xxx_weights` will randomly initialize a model.

## Device

Each Inferencer instance is bound to a device.
By default, the best device is automatically decided by [MMEngine](https://github.com/open-mmlab/mmengine/). You can also alter the device by specifying the `device` argument. Refer to [torch.device](torch.device) for all the supported forms.

## Batch Inference

You can set the batch size by setting the `batch_size` argument. The default batch size is 1.

## Return Value

By default, each `Inferecner` returns the prediction results in a dictionary format.

- `visualization` contains the visualized predictions. But it's an empty list by default unless `return_vis=True`.

- `predictions` contains the predictions results in a json-serializable format. As presented below, the keys are slightly different depending on the task type.

  **TextDetInferencer**

  ```python
  {
      'predictions' : [
        # Each instance corresponds to an input image
        {
          'polygons': [...],  # 2d list of len (N,) in the format of [x1, y1, x2, y2, ...]
          'bboxes': [...],  # 2d list of shape (N, 4), in the format of [min_x, min_y, max_x, max_y]
          'scores': [...]  # list of float, len (N, )
        },
      ]
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  ```

  ### TextRecInferencer

  ```python
  {
      'predictions' : [
        # Each instance corresponds to an input image
        {
          'text': '...',  # a string
          'scores': 0.1,  # a float
        },
        ...
      ]
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  ```

  **TextSpottingInferencer**

  ```python
  {
      'predictions' : [
        # Each instance corresponds to an input image
        {
          'polygons': [...],  # 2d list of len (N,) in the format of [x1, y1, x2, y2, ...]
          'bboxes': [...],  # 2d list of shape (N, 4), in the format of [min_x, min_y, max_x, max_y]
          'scores': [...]  # list of float, len (N, )
          'texts': ['...',]  # list of texts, len (N, )
        },
      ]
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  ```

  **KIEInferencer**

  ```python
  {
      'predictions' : [
        # Each instance corresponds to an input image
        {
          'labels': [...],  # node label, len (N,)
          'scores': [...],  # node scores, len (N, )
          'edge_scores': [...],  # edge scores, shape (N, N)
          'edge_labels': [...],  # edge labels, shape (N, N)
        },
      ]
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  ```

If you wish to get the raw outputs from the model, you can set `return_datasamples` to `True` to get the original [DataSample](structures.md), which will be stored in `predictions`.

## Dumping Results

Apart from obtaining predictions from the return value, you can also export the predictions/visualization to files by setting `out_dir` and `save_pred`/`save_vis` arguments. Assuming `out_dir` is `outputs`, the files will be organized as follows:

```text
outputs
├── preds
│   └── img_1.json
└── vis
    └── img_1.jpg
```

The filename of each file is the same as the corresponding input image filename. If the input image is an array, the filename will be a number starting from 0.

## API Arguments

The API has an extensive list of arguments that you can use. The following tables are for the python interface.

**MMOCRInferencer():**

| Arguments     | Type                                    | Default | Description                                                                                                                                   |
| ------------- | --------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `det`         | see [Weights](../modelzoo.html#weights) | None    | Pretrained text detection algorithm. It's the path to the config file or the model name defined in metafile.                                  |
| `det_weights` | str                                     | None    | Path to the custom checkpoint file of the selected det model. If it is not specified and "det" is a model name of metafile, the weights will be loaded from metafile. |
| `rec`         | see [Weights](../modelzoo.html#weights) | None    | Pretrained text recognition algorithm. It’s the path to the config file or the model name defined in metafile.                                |
| `rec_weights` | str                                     | None    | Path to the custom checkpoint file of the selected rec model. If it is not specified and “rec” is a model name of metafile, the weights will be loaded from metafile. |
| `kie` \[1\]   | see [Weights](../modelzoo.html#weights) | None    | Pretrained key information extraction algorithm. It’s the path to the config file or the model name defined in metafile.                      |
| `kie_weights` | str                                     | None    | Path to the custom checkpoint file of the selected kie model. If it is not specified and “kie” is a model name of metafile, the weights will be loaded from metafile. |
| `device`      | str                                     | None    | Device used for inference, accepting all allowed strings by `torch.device`. E.g., 'cuda:0' or 'cpu'. If None, the available device will be automatically used. Defaults to None. |

\[1\]: `kie` is only effective when both text detection and recognition models are specified.

### \_\_call\_\_*()*

| Arguments            | Type                    | Default      | Description                                                                                      |
| -------------------- | ----------------------- | ------------ | ------------------------------------------------------------------------------------------------ |
| `inputs`             | str/list/tuple/np.array | **required** | It can be a path to an image/a folder, an np array or a list/tuple (with img paths or np arrays) |
| `return_datasamples` | bool                    | False        | Whether to return results as DataSamples. If False, the results will be packed into a dict.      |
| `batch_size`         | int                     | 1            | Inference batch size.                                                                            |
| `return_vis`         | bool                    | False        | Whether to return the visualization result.                                                      |
| `print_result`       | bool                    | False        | Whether to print the inference result to the console.                                            |
| `wait_time`          | float                   | 0            | The interval of show(s).                                                                         |
| `out_dir`            | str                     | `results/`   | Output directory of results.                                                                     |
| `save_vis`           | bool                    | False        | Whether to save the visualization results to `out_dir`.                                          |
| `save_pred`          | bool                    | False        | Whether to save the inference results to `out_dir`.                                              |
