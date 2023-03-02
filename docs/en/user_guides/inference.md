# Inference

In OpenMMLab, all the inference operations are unified into a new interface - `Inferencer`. `Inferencer` is designed to expose a neat and simple API to users, and shares very similar interface across different OpenMMLab libraries.

In MMOCR, Inferencers are constructed in different levels of task abstraction.

- Standard Inferencer: Following OpenMMLab's convention, each fundamental task in MMOCR has a standard Inferencer, namely `TextDetInferencer`, `TextRecInferencer`, `TextSpottingInferencer`, and `KIEInferencer`. They are designed to perform inference on a single task, and can be chained together to perform inference on a series of tasks. They also share very similar interface, have standard input/output protocol, and overall follow the OpenMMLab design.
- **MMOCRInferencer**: We also provide `MMOCRInferencer`, a convenient inference interface only designed for MMOCR. It encapsulates and chains all the Inferencers in MMOCR, so users can use this Inferencer to perform a series of tasks on an image and directly get the final result in an end-to-end manner. *However, it has a relatively different interface from other standard Inferencers, and some of standard Inferencer functionalities might be sacrificed for the sake of simplicity.*

For new users, we recommend using **MMOCRInferencer** to test out different combinations of models.

If you are a developer and wish to integrate the models into your own project, we recommend using **standard Inferencers**, as they are more flexible and standardized, equipped with full functionalities.

## Basic Usage

`````{tabs}

````{group-tab} MMOCRInferencer

As of now, `MMOCRInferencer` can perform inference on the following tasks:

- Text detection
- Text recognition
- OCR (text detection + text recognition)
- Key information extraction (text detection + text recognition + key information extraction)
- *OCR (text spotting)* (coming soon)

For convenience, `MMOCRInferencer` provides both Python and command line interfaces. For example, if you want to perform OCR inference on `demo/demo_text_ocr.jpg` with `DBNet` as the text detection model and `CRNN` as the text recognition model, you can simply run the following command:

::::{tabs}

:::{code-tab} python
>>> from mmocr.apis import MMOCRInferencer
>>> # Load models into memory
>>> ocr = MMOCRInferencer(det='DBNet', rec='SAR')
>>> # Perform inference
>>> ocr('demo/demo_text_ocr.jpg', show=True)
:::

:::{code-tab} bash
python tools/infer.py demo/demo_text_ocr.jpg --det DBNet --rec SAR --show
:::
::::

The resulting OCR output will be displayed in a new window:

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/220563262-e9c1ab52-9b96-4d9c-bcb6-f55ff0b9e1be.png" height="250"/>
</div>

```{note}
If you are running MMOCR on a server without GUI or via SSH tunnel with X11 forwarding disabled, the `show` option will not work. However, you can still save visualizations to files by setting `out_dir` and `save_vis=True` arguments. Read [Dumping Results](#dumping-results) for details.
```

Depending on the initialization arguments, `MMOCRInferencer` can run in different modes. For example, it can run in KIE mode if it is initialized with `det`, `rec` and `kie` specified.

::::{tabs}

:::{code-tab} python
>>> kie = MMOCRInferencer(det='DBNet', rec='SAR', kie='SDMGR')
>>> kie('demo/demo_kie.jpeg', show=True)
:::

:::{code-tab} bash
python tools/infer.py demo/demo_kie.jpeg --det DBNet --rec SAR --kie SDMGR --show
:::

::::

The output image should look like this:

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/220569700-fd4894bc-f65a-405e-95e7-ebd2d614aedd.png" height="250"/>
</div>
<br />

You may have found that the Python interface and the command line interface of `MMOCRInferencer` are very similar. The following sections will use the Python interface as an example to introduce the usage of `MMOCRInferencer`. For more information about the command line interface, please refer to [Command Line Interface](#command-line-interface).

````

````{group-tab} Standard Inferencer

In general, all the standard Inferencers across OpenMMLab share a very similar interface. The following example shows how to use `TextDetInferencer` to perform inference on a single image.

```python
>>> from mmocr.apis import TextDetInferencer
>>> # Load models into memory
>>> inferencer = TextDetInferencer(model='DBNet')
>>> # Inference
>>> inferencer('demo/demo_text_ocr.jpg', show=True)
```

The visualization result should look like:

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/221418215-2431d0e9-e16e-4deb-9c52-f8b86801706a.png" height="250"/>
</div>

````

`````

## Initialization

Each Inferencer must be initialized with a model. You can also choose the inference device during initialization.

### Model Initialization

`````{tabs}

````{group-tab} MMOCRInferencer

For each task, `MMOCRInferencer` takes two arguments in the form of `xxx` and `xxx_weights` (e.g. `det` and `det_weights`) for initialization, and there are many ways to initialize a model for inference. We will take `det` and `det_weights` as an example to illustrate some typical ways to initialize a model.

- To infer with MMOCR's pre-trained model, passing its name to the argument `det` can work. The weights will be automatically downloaded and loaded from OpenMMLab's model zoo. Check [Weights](../modelzoo.md#weights) for available model names.

  ```python
  >>> MMOCRInferencer(det='DBNet')
  ```

- To load custom config and weight, you can pass the path to the config file to `det` and the path to the weight to `det_weights`.

  ```python
  >>> MMOCRInferencer(det='path/to/dbnet_config.py', det_weights='path/to/dbnet.pth')
  ```

You may click on the "Standard Inferencer" tab to find more initialization methods.

````

````{group-tab} Standard Inferencer

Every standard `Inferencer` accepts two parameters, `model` and `weights`. (In `MMOCRInferencer`, they are referred to as `xxx` and `xxx_weights`)

- `model` takes either the name of a model, or the path to a config file as input. The name of a model is obtained from the model's metafile ([Example](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/metafile.yml)) indexed from [model-index.yml](https://github.com/open-mmlab/mmocr/blob/1.x/model-index.yml). You can find the list of available weights [here](../modelzoo.md#weights).

- `weights` accepts the path to a weight file.

<br />

There are various ways to initialize a model.

- To infer with MMOCR's pre-trained model,  you can pass its name to `model`. The weights will be automatically downloaded and loaded from OpenMMLab's model zoo.

  ```python
  >>> from mmocr.apis import TextDetInferencer
  >>> inferencer = TextDetInferencer(model='DBNet')
  ```

  ```{note}
  The model type must match the Inferencer type.
  ```

  You can load another weight by passing its path/url to `weights`.

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

- Passing config file to `model` without specifying `weight` will result in a randomly initialized model.

````
`````

### Device

Each Inferencer instance is bound to a device.
By default, the best device is automatically decided by [MMEngine](https://github.com/open-mmlab/mmengine/). You can also alter the device by specifying the `device` argument. For example, you can use the following code to create an Inferencer on GPU 1.

`````{tabs}

````{group-tab} MMOCRInferencer

```python
>>> inferencer = MMOCRInferencer(det='DBNet', device='cuda:1')
```

````

````{group-tab} Standard Inferencer

```python
>>> inferencer = TextDetInferencer(model='DBNet', device='cuda:1')
```

````

`````

To create an Inferencer on CPU:

`````{tabs}

````{group-tab} MMOCRInferencer

```python
>>> inferencer = MMOCRInferencer(det='DBNet', device='cpu')
```

````

````{group-tab} Standard Inferencer

```python
>>> inferencer = TextDetInferencer(model='DBNet', device='cpu')
```

````

`````

Refer to [torch.device](torch.device) for all the supported forms.

## Inference

Once the Inferencer is initialized, you can directly pass in the raw data to be inferred and get the inference results from return values.

### Input

`````{tabs}

````{tab} MMOCRInferencer / TextDetInferencer / TextRecInferencer / TextSpottingInferencer

Input can be either of these types:

- str: Path/URL to the image.

  ```python
  >>> inferencer('demo/demo_text_ocr.jpg')
  ```

- array: Image in numpy array. It should be in BGR order.

  ```python
  >>> import mmcv
  >>> array = mmcv.imread('demo/demo_text_ocr.jpg')
  >>> inferencer(array)
  ```

- list: A list of basic types above. Each element in the list will be processed separately.

  ```python
  >>> inferencer(['img_1.jpg', 'img_2.jpg])
  >>> # You can even mix the types
  >>> inferencer(['img_1.jpg', array])
  ```

- str: Path to the directory. All images in the directory will be processed.

  ```python
  >>> inferencer('tests/data/det_toy_dataset/imgs/test/')
  ```

````

````{tab} KIEInferencer

Input can be a dict or list[dict], where each dictionary contains
following keys:

- `img` (str or ndarray): Path to the image or the image itself. If KIE Inferencer is used in no-visual mode, this key is not required.
If it's an numpy array, it should be in BGR order.
- `img_shape` (tuple(int, int)): Image shape in (H, W). Only required when KIE Inferencer is used in no-visual mode and no `img` is provided.
- `instances` (list[dict]): A list of instances.

Each `instance` looks like the following:

```python
{
    # A nested list of 4 numbers representing the bounding box of
    # the instance, in (x1, y1, x2, y2) order.
    "bbox": np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                    dtype=np.int32),

    # List of texts.
    "texts": ['text1', 'text2', ...],
}
```

````
`````

### Output

By default, each `Inferencer` returns the prediction results in a dictionary format.

- `visualization` contains the visualized predictions. But it's an empty list by default unless `return_vis=True`.

- `predictions` contains the predictions results in a json-serializable format. As presented below, the contents are slightly different depending on the task type.

  `````{tabs}

  :::{group-tab} MMOCRInferencer

  ```python
  {
      'predictions' : [
        # Each instance corresponds to an input image
        {
          'det_polygons': [...],  # 2d list of length (N,), format: [x1, y1, x2, y2, ...]
          'det_scores': [...],  # float list of length (N,)
          'det_bboxes': [...],   # 2d list of shape (N, 4), format: [min_x, min_y, max_x, max_y]
          'rec_texts': [...],  # str list of length (N,)
          'rec_scores': [...],  # float list of length (N,)
          'kie_labels': [...],  # node labels, length (N, )
          'kie_scores': [...],  # node scores, length (N, )
          'kie_edge_scores': [...],  # edge scores, shape (N, N)
          'kie_edge_labels': [...]  # edge labels, shape (N, N)
        },
        ...
      ],
      'visualization' : [
        array(..., dtype=uint8),
      ]
  }
  ```

  :::

  :::{group-tab} Standard Inferencer

  ````{tabs}
  ```{code-tab} python TextDetInferencer

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

  ```{code-tab} python TextRecInferencer
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

  ```{code-tab} python TextSpottingInferencer
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

  ```{code-tab} python KIEInferencer
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
  ````

  :::

  `````

If you wish to get the raw outputs from the model, you can set `return_datasamples` to `True` to get the original [DataSample](structures.md), which will be stored in `predictions`.

### Dumping Results

Apart from obtaining predictions from the return value, you can also export the predictions/visualizations to files by setting `out_dir` and `save_pred`/`save_vis` arguments.

```python
>>> inferencer('img_1.jpg', out_dir='outputs/', save_pred=True, save_vis=True)
```

Results in the directory structure like:

```text
outputs
├── preds
│   └── img_1.json
└── vis
    └── img_1.jpg
```

The filename of each file is the same as the corresponding input image filename. If the input image is an array, the filename will be a number starting from 0.

### Batch Inference

You can customize the batch size by setting `batch_size`. The default batch size is 1.

## API

Here are extensive lists of parameters that you can use.

````{tabs}

```{group-tab} MMOCRInferencer

**MMOCRInferencer.\_\_init\_\_():**

| Arguments     | Type                                                 | Default | Description                                                                                                                                                                      |
| ------------- | ---------------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `det`         | str or [Weights](../modelzoo.html#weights), optional | None    | Pretrained text detection algorithm. It's the path to the config file or the model name defined in metafile.                                                                     |
| `det_weights` | str, optional                                        | None    | Path to the custom checkpoint file of the selected det model. If it is not specified and "det" is a model name of metafile, the weights will be loaded from metafile.            |
| `rec`         | str or [Weights](../modelzoo.html#weights), optional | None    | Pretrained text recognition algorithm. It’s the path to the config file or the model name defined in metafile.                                                                   |
| `rec_weights` | str, optional                                        | None    | Path to the custom checkpoint file of the selected rec model. If it is not specified and “rec” is a model name of metafile, the weights will be loaded from metafile.            |
| `kie` \[1\]   | str or [Weights](../modelzoo.html#weights), optional | None    | Pretrained key information extraction algorithm. It’s the path to the config file or the model name defined in metafile.                                                         |
| `kie_weights` | str, optional                                        | None    | Path to the custom checkpoint file of the selected kie model. If it is not specified and “kie” is a model name of metafile, the weights will be loaded from metafile.            |
| `device`      | str, optional                                        | None    | Device used for inference, accepting all allowed strings by `torch.device`. E.g., 'cuda:0' or 'cpu'. If None, the available device will be automatically used. Defaults to None. |

\[1\]: `kie` is only effective when both text detection and recognition models are specified.

**MMOCRInferencer.\_\_call\_\_()**

| Arguments            | Type                    | Default      | Description                                                                                      |
| -------------------- | ----------------------- | ------------ | ------------------------------------------------------------------------------------------------ |
| `inputs`             | str/list/tuple/np.array | **required** | It can be a path to an image/a folder, an np array or a list/tuple (with img paths or np arrays) |
| `return_datasamples` | bool                    | False        | Whether to return results as DataSamples. If False, the results will be packed into a dict.      |
| `batch_size`         | int                     | 1            | Inference batch size.                                                                            |
| `return_vis`         | bool                    | False        | Whether to return the visualization result.                                                      |
| `print_result`       | bool                    | False        | Whether to print the inference result to the console.                                            |
| `show`               | bool                    | False        | Whether to display the visualization results in a popup window.                                  |
| `wait_time`          | float                   | 0            | The interval of show(s).                                                                         |
| `out_dir`            | str                     | `results/`   | Output directory of results.                                                                     |
| `save_vis`           | bool                    | False        | Whether to save the visualization results to `out_dir`.                                          |
| `save_pred`          | bool                    | False        | Whether to save the inference results to `out_dir`.                                              |

```

```{group-tab} Standard Inferencer

**Inferencer.\_\_init\_\_():**

| Arguments | Type                                                 | Default | Description                                                                                                                                                                      |
| --------- | ---------------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`   | str or [Weights](../modelzoo.html#weights), optional | None    | Path to the config file or the model name defined in metafile.                                                                                                                   |
| `weights` | str, optional                                        | None    | Path to the custom checkpoint file of the selected det model. If it is not specified and "det" is a model name of metafile, the weights will be loaded from metafile.            |
| `device`  | str, optional                                        | None    | Device used for inference, accepting all allowed strings by `torch.device`. E.g., 'cuda:0' or 'cpu'. If None, the available device will be automatically used. Defaults to None. |

**Inferencer.\_\_call\_\_()**

| Arguments            | Type                    | Default      | Description                                                                                                      |
| -------------------- | ----------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------- |
| `inputs`             | str/list/tuple/np.array | **required** | It can be a path to an image/a folder, an np array or a list/tuple (with img paths or np arrays)                 |
| `return_datasamples` | bool                    | False        | Whether to return results as DataSamples. If False, the results will be packed into a dict.                      |
| `batch_size`         | int                     | 1            | Inference batch size.                                                                                            |
| `progress_bar`       | bool                    | True         | Whether to show a progress bar.                                                                                  |
| `return_vis`         | bool                    | False        | Whether to return the visualization result.                                                                      |
| `print_result`       | bool                    | False        | Whether to print the inference result to the console.                                                            |
| `show`               | bool                    | False        | Whether to display the visualization results in a popup window.                                                  |
| `wait_time`          | float                   | 0            | The interval of show(s).                                                                                         |
| `draw_pred`          | bool                    | True         | Whether to draw predicted bounding boxes. *Only applicable on `TextDetInferencer` and `TextSpottingInferencer`.* |
| `out_dir`            | str                     | `results/`   | Output directory of results.                                                                                     |
| `save_vis`           | bool                    | False        | Whether to save the visualization results to `out_dir`.                                                          |
| `save_pred`          | bool                    | False        | Whether to save the inference results to `out_dir`.                                                              |

```
````

## Command Line Interface

```{note}
This section is only applicable to `MMOCRInferencer`.
```

You can use `tools/infer.py` to perform inference through `MMOCRInferencer`.
Its general usage is as follows:

```bash
python tools/infer.py INPUT_PATH [--det DET] [--det-weights ...] ...
```

where `INPUT_PATH` is a required field, which should be a path to an image or a folder. Command-line parameters follow the mapping relationship with the Python interface parameters as follows:

- To convert the Python interface parameters to the command line ones, you need to add two `--` in front of the Python interface parameters, and replace the underscore `_` with the hyphen `-`. For example, `out_dir` becomes `--out-dir`.
- For boolean type parameters, putting the parameter in the command is equivalent to specifying it as True. For example, `--show` will specify the `show` parameter as True.

In addition, the command line will not display the inference result by default. You can use the `--print-result` parameter to view the inference result.

Here is an example:

```bash
python tools/infer.py demo/demo_text_ocr.jpg --det DBNet --rec SAR --show --print-result
```

Running this command will give the following result:

```bash
{'predictions': [{'rec_texts': ['CBank', 'Docbcba', 'GROUP', 'MAUN', 'CROBINSONS', 'AOCOC', '916M3', 'BOO9', 'Oven', 'BRANDS', 'ARETAIL', '14', '70<UKN>S', 'ROUND', 'SALE', 'YEAR', 'ALLY', 'SALE', 'SALE'],
'rec_scores': [0.9753464579582214, ...], 'det_polygons': [[551.9930285844646, 411.9138765335083, 553.6153911653112,
383.53195309638977, 620.2410061195247, 387.33785033226013, 618.6186435386782, 415.71977376937866], ...], 'det_scores': [0.8230461478233337, ...]}]}
```
