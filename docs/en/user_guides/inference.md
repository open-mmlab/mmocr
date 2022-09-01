# Inference

We provide an easy-to-use API for the demo and application purpose in [ocr.py](/mmocr/ocr.py) script.

The API can be called through command line (CL) or by calling it from another python script.
It exposes all the models in MMOCR to API as individual modules that can be called and chained together.

```{warning}
This interface is being refactored is much likely to be changed in the upcoming release.
```

## Example 1: Text Detection

<div align="center">
    <img src="https://user-images.githubusercontent.com/24622904/187825864-8ead5acb-c3c5-443b-bd90-3f4b188fa315.jpg"  height="250"/>
</div>

**Instruction:** Perform detection inference on an image with the TextSnake recognition model, export the result in a json file (default) and save the visualization file.

- CL interface:

```shell
python mmocr/ocr.py demo/demo_text_det.jpg --det TextSnake --img-out-dir demo/
```

- Python interface:

```python
from mmocr.ocr import MMOCR

# Load models into memory
ocr = MMOCR(det='TextSnake')

# Inference
results = ocr.readtext('demo/demo_text_det.jpg', img_out_dir='demo/')
```

## Example 2: Text Detection + Recognition

<div align="center">
    <img src="https://user-images.githubusercontent.com/24622904/187825445-d30cbfa6-5549-4358-97fe-245f08f4ed94.jpg" height="250"/>
</div>

**Instruction:** Perform ocr (det + recog) inference on the demo/demo_text_det.jpg image with the DB_r18 detection model and CRNN recognition model, print the result in the terminal and show the visualization.

- CL interface:

```shell
python mmocr/ocr.py --det DB_r18 --recog CRNN demo/demo_text_ocr.jpg --print-result --show
```

```{note}

When calling the script from the command line, the script assumes configs are saved in the `configs/` folder. User can customize the directory by specifying the value of `config_dir`.

```

- Python interface:

```python
from mmocr.ocr import MMOCR

# Load models into memory
ocr = MMOCR()

# Inference
results = ocr.readtext('demo/demo_text_ocr.jpg', print_result=True, show=True)
```

## Example 3: Text Detection + Recognition + Key Information Extraction

<div align="center">
    <img src="https://user-images.githubusercontent.com/24622904/187825451-6b043df9-10f7-4656-a528-45fe043df92b.jpg" height="250"/>
</div>

**Instruction:** Perform end-to-end ocr (det + recog) inference first with DB_r18 detection model and CRNN recognition model, then run KIE inference with SDMGR model on the ocr result and show the visualization.

- CL interface:

```shell
python mmocr/ocr.py demo/demo_kie.jpeg  --det DB_r18 --recog CRNN --kie SDMGR --print-result --show
```

```{note}

Note: When calling the script from the command line, the script assumes configs are saved in the `configs/` folder. User can customize the directory by specifying the value of `config_dir`.

```

- Python interface:

```python
from mmocr.ocr import MMOCR

# Load models into memory
ocr = MMOCR(det='DB_r18', recog='CRNN', kie='SDMGR')

# Inference
results = ocr.readtext('demo/demo_kie.jpeg', print_result=True, show=True)
```

## API Arguments

The API has an extensive list of arguments that you can use. The following tables are for the python interface.

**MMOCR():**

| Arguments      | Type                  | Default  | Description                                                                                          |
| -------------- | --------------------- | -------- | ---------------------------------------------------------------------------------------------------- |
| `det`          | see [models](#models) | None     | Text detection algorithm                                                                             |
| `recog`        | see [models](#models) | None     | Text recognition algorithm                                                                           |
| `kie` \[1\]    | see [models](#models) | None     | Key information extraction algorithm                                                                 |
| `config_dir`   | str                   | configs/ | Path to the config directory where all the config files are located                                  |
| `det_config`   | str                   | None     | Path to the custom config file of the selected det model                                             |
| `det_ckpt`     | str                   | None     | Path to the custom checkpoint file of the selected det model                                         |
| `recog_config` | str                   | None     | Path to the custom config file of the selected recog model                                           |
| `recog_ckpt`   | str                   | None     | Path to the custom checkpoint file of the selected recog model                                       |
| `kie_config`   | str                   | None     | Path to the custom config file of the selected kie model                                             |
| `kie_ckpt`     | str                   | None     | Path to the custom checkpoint file of the selected kie model                                         |
| `device`       | str                   | None     | Device used for inference, accepting all allowed strings by `torch.device`. E.g., 'cuda:0' or 'cpu'. |

\[1\]: `kie` is only effective when both text detection and recognition models are specified.

```{note}

User can use default pretrained models by specifying `det` and/or `recog`, which is equivalent to specifying their corresponding `*_config` and `*_ckpt`. However, manually specifying `*_config` and `*_ckpt` will always override values set by `det` and/or `recog`. Similar rules also apply to `kie`, `kie_config` and `kie_ckpt`.

```

### readtext()

| Arguments      | Type                    | Default      | Description                                                            |
| -------------- | ----------------------- | ------------ | ---------------------------------------------------------------------- |
| `img`          | str/list/tuple/np.array | **required** | img, folder path, np array or list/tuple (with img paths or np arrays) |
| `img_out_dir`  | str                     | None         | Output directory of images.                                            |
| `show`         | bool                    | False        | Whether to show the result visualization on screen                     |
| `print_result` | bool                    | False        | Whether to show the result for each image                              |

All arguments are the same for the cli, all you need to do is add 2 hyphens at the beginning of the argument and replace underscores by hyphens.
(*Example:* `img_out_dir` becomes `--img-out-dir`)

For bool type arguments, putting the argument in the command stores it as true.
(*Example:* `python mmocr/demo/ocr.py --det DB_r18 demo/demo_text_det.jpg --print_result`
means that `print_result` is set to `True`)

## Models

**Text detection:**

| Name          |                                                                         Reference                                                                         |
| ------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DB_r18        |            [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#real-time-scene-text-detection-with-differentiable-binarization)            |
| DB_r50        |            [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#real-time-scene-text-detection-with-differentiable-binarization)            |
| DBPP_r50      |                                        [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#dbnetpp)                                        |
| DRRG          |                                         [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#drrg)                                          |
| FCE_IC15      |             [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)             |
| FCE_CTW_DCNv2 |             [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)             |
| MaskRCNN_CTW  |                                      [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#mask-r-cnn)                                       |
| MaskRCNN_IC15 |                                      [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#mask-r-cnn)                                       |
| PANet_CTW     | [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#efficient-and-accurate-arbitrary-shaped-text-detection-with-pixel-aggregation-network) |
| PANet_IC15    | [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#efficient-and-accurate-arbitrary-shaped-text-detection-with-pixel-aggregation-network) |
| PS_CTW        |                                        [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#psenet)                                         |
| PS_IC15       |                                        [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#psenet)                                         |
| TextSnake     |                                       [link](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html#textsnake)                                       |

**Text recognition:**

| Name |                                                                                            Reference                                                                                            |
| ---- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| CRNN | [link](https://mmocr.readthedocs.io/en/dev-1.x/textrecog_models.html#an-end-to-end-trainable-neural-network-for-image-based-sequence-recognition-and-its-application-to-scene-text-recognition) |

**Key information extraction:**

| Name  |                                                              Reference                                                               |
| ----- | :----------------------------------------------------------------------------------------------------------------------------------: |
| SDMGR | [link](https://mmocr.readthedocs.io/en/dev-1.x/kie_models.html#spatial-dual-modality-graph-reasoning-for-key-information-extraction) |

## Additional info

- To perform det + recog inference (end2end ocr), both the `det` and `recog` arguments must be defined.
- To perform only detection set the `recog` argument to `None`.
- To perform only recognition set the `det` argument to `None`.

If you have any suggestions for new features, feel free to open a thread or even PR :)
