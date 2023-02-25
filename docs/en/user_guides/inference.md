# Inference

We provide an easy-to-use API - `MMOCRInferencer`, for the demo purpose in [ocr.py](/mmocr/ocr.py) script. It can perform inference on following tasks:

- Text detection
- Text recognition
- OCR (text detection + text recognition)
- Key information extraction (text detection + text recognition + key information extraction)
- *OCR (text spotting)* (coming soon)

These tasks are performed by using one or several task-specific [Inferencers](../basic_concepts/inferencers.md). `MMOCRInferencer` encapsulates and chains all the Inferencers in MMOCR, so users can use this Inferencer to perform a series of tasks on an image and directly get the final result in an end-to-end manner.

The following sections will guide you through some basic usages of `MMOCRInferencer`.

## Basic Usage

Assuming that we want to perform OCR on `demo/demo_text_ocr.jpg`, using `DBNet` as text detection model and `CRNN` as text recognition model. We can use the following command to perform the inference:

```python
>>> from mmocr.apis import MMOCRInferencer
>>> # Load models into memory
>>> ocr = MMOCRInferencer(det='DBNet', rec='SAR')
>>> # Inference
>>> ocr('demo/demo_text_ocr.jpg', show=True)
```

The OCR result will be visualized in a new window:

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/220563262-e9c1ab52-9b96-4d9c-bcb6-f55ff0b9e1be.png" height="250"/>
</div>

```{note}
If you are running MMOCR on a server without GUI or via SSH tunnel with X11 forwarding off, the `show` option will not work. You can still save visualizations to files by setting `out_dir` and `save_vis=True` arguments. Read [Get Results](#get-results) for details.
```

Depending on the initialization arguments, `MMOCRInferencer` can run in different modes. For example, it can run in KIE mode if it is initialized with `det`, `rec` and `kie` specified.

```python
>>> kie = MMOCRInferencer(det='DBNet', rec='SAR', kie='SDMGR')
>>> kie('demo/demo_kie.jpeg', show=True)
```

Which should give you an image like this:

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/220569700-fd4894bc-f65a-405e-95e7-ebd2d614aedd.png" height="250"/>
</div>
<br />

`MMOCRInferencer` accepts many types of inputs. It can be an numpy array or the path/url to an image. If you have several inputs, a list of them is acceptable:

```python
>>> import mmcv
>>> # Load the image as a numpy array
>>> np_img = mmcv.imread('tests/data/det_toy_dataset/imgs/test/img_1.jpg')
>>> # Passing a list of inputs. Mixing numpy array and path string is fine
>>> ocr([np_img,  'tests/data/det_toy_dataset/imgs/test/img_10.jpg'], show=True)
```

Sometimes you may want to iterate over a directory where all the images are stored. Just pass the directory path to `MMOCRInferencer`:

```python
>>> ocr('tests/data/det_toy_dataset/imgs/test/', show=True)
```

## Model Initialization

For each task, `MMOCRInferencer` takes two arguments in the form of `xxx` and `xxx_weights` (e.g. `det` and `det_weights`) for initialization, and there are many ways to initialize a model for inference. We will take `det` and `det_weights` as an example to illustrate the way to initialize a model.

- To infer with MMOCR's pre-trained model, passing its name to the argument `det` can work. The weights will be automatically downloaded and loaded from OpenMMLab's model zoo. Check [Weights](../modelzoo.md#weights) for available model names.

  ```python
  >>> MMOCRInferencer(det='DBNet')
  ```

  To load the custom weight, you can also pass the path/url to `det_weights`.

  ```python
  >>> MMOCRInferencer(det='DBNet', det_weights='path/to/dbnet.pth')
  ```

- To load custom config and weight, you can pass the path to the config file to `det` and the path to the weight to `det_weights`.

  ```python
  >>> MMOCRInferencer(det='path/to/dbnet_config.py', det_weights='path/to/dbnet.pth')
  ```

- If you have a weight trained on [MMEngine](https://github.com/open-mmlab/mmengine/), specifying `xxx_weights` only is also fine - the config will be automatically loaded from the weight.

  ```python
  >>> # It will raise an error if the config file cannot be found in the weight
  >>> MMOCRInferencer(det_weights='path/to/dbnet.pth')
  ```

- Passing config file to `xxx` without specifying the weight path `xxx_weights` will randomly initialize a model.

## Device

Each Inferencer instance is bound to a device.
By default, the best device is automatically decided by [MMEngine](https://github.com/open-mmlab/mmengine/). You can also alter the device by specifying the `device` argument. Refer to [torch.device](torch.device) for all the supported forms.

## Batch Inference

You can set the batch size by setting the `batch_size` argument. The default batch size is 1.

## Get Results

In Python interface, `MMOCRInferencer` returns predictions in a dictionary format. The keys starts with the task name, i.e. `det`, `rec` and `kie`; the values are the corresponding predictions. Depending on the actual task that MMOCRInferencer is running, the return values may be a subset of the following:

```python
{
    'predictions' : [{
        'det_polygons': [...],
        'det_scores': [...]
        'rec_texts': [...],
        'rec_scores': [...],
        'kie_labels': [...],
        'kie_scores': [...],
        'kie_edge_scores': [...],
        'kie_edge_labels': [...]
    },]
    'visualization' : [array(..., dtype=uint8),]
}
```

`predictions` is a list of dictionaries. Each dictionary formats the inference result of the corresponding image. Similarly, `visualization` is a list of numpy arrays, each array corresponds to the visualization of an image.

```{note}
The visualization result will only be returned when `return_vis=True`.
```

Apart from obtaining predictions from the return value, you can also export the predictions/visualization to files by setting `out_dir` and `save_pred`/`save_vis` arguments. Assuming `out_dir` is `outputs`, the files will be organized as follows:

```text
outputs
├── preds
│   └── img_1.json
└── vis
    └── img_1.jpg
```

The filename of each file is the same as the corresponding input image filename. If the input image is an array, the filename will be a number starting from 0.

## CLI Interface

`MMOCRInferencer` supports both CLI and Python interface. All arguments are the same for the CLI, all you need to do is add 2 hyphens at the beginning of the argument and replace underscores by hyphens.
(*Example:* `out_dir` becomes `--out-dir`)

For bool type arguments, putting the argument in the command stores it as true.

For example, the [first example](#simple-usage) can be run in CLI as:

```bash
python tools/infer.py demo/demo_text_ocr.jpg --det DBNet --rec CRNN --show
```

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
