# Demo

We provide an easy-to-use API for the demo and application purpose in [ocr.py](https://github.com/open-mmlab/mmocr/blob/main/mmocr/utils/ocr.py) script.

The API can be called through command line (CL) or by calling it from another python script.
It exposes all the models in MMOCR to API as individual modules that can be called and chained together. [Tesseract](https://tesseract-ocr.github.io/) is integrated as a text detector and/or recognizer in the task pipeline.

______________________________________________________________________

## Example 1: Text Detection

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/text_det_pred.jpg"/><br>
</div>
<br>

**Instruction:** Perform detection inference on an image with the TextSnake recognition model, export the result in a json file (default) and save the visualization file.

- CL interface:

```shell
python mmocr/utils/ocr.py demo/demo_text_det.jpg --output demo/det_out.jpg --det TextSnake --recog None --export demo/
```

- Python interface:

```python
from mmocr.utils.ocr import MMOCR

# Load models into memory
ocr = MMOCR(det='TextSnake', recog=None)

# Inference
results = ocr.readtext('demo/demo_text_det.jpg', output='demo/det_out.jpg', export='demo/')
```

## Example 2: Text Recognition

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/text_recog_pred.jpg"/><br>
</div>
<br>

**Instruction:** Perform batched recognition inference on a folder with hundreds of image with the CRNN_TPS recognition model and save the visualization results in another folder.
*Batch size is set to 10 to prevent out of memory CUDA runtime errors.*

- CL interface:

```shell
python mmocr/utils/ocr.py %INPUT_FOLDER_PATH% --det None --recog CRNN_TPS --batch-mode --single-batch-size 10 --output %OUPUT_FOLDER_PATH%
```

- Python interface:

```python
from mmocr.utils.ocr import MMOCR

# Load models into memory
ocr = MMOCR(det=None, recog='CRNN_TPS')

# Inference
results = ocr.readtext(%INPUT_FOLDER_PATH%, output = %OUTPUT_FOLDER_PATH%, batch_mode=True, single_batch_size = 10)
```

## Example 3: Text Detection + Recognition

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/demo_ocr_pred.jpg"/><br>
</div>
<br>

**Instruction:** Perform ocr (det + recog) inference on the demo/demo_text_det.jpg image with the PANet_IC15 (default) detection model and SAR (default) recognition model, print the result in the terminal and show the visualization.

- CL interface:

```shell
python mmocr/utils/ocr.py demo/demo_text_ocr.jpg --print-result --imshow
```

```{note}

When calling the script from the command line, the script assumes configs are saved in the `configs/` folder. User can customize the directory by specifying the value of `config_dir`.

```

- Python interface:

```python
from mmocr.utils.ocr import MMOCR

# Load models into memory
ocr = MMOCR()

# Inference
results = ocr.readtext('demo/demo_text_ocr.jpg', print_result=True, imshow=True)
```

______________________________________________________________________

## Example 4: Text Detection + Recognition + Key Information Extraction

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/demo_kie_pred.png"/><br>
</div>
<br>

**Instruction:** Perform end-to-end ocr (det + recog) inference first with PS_CTW detection model and SAR recognition model, then run KIE inference with SDMGR model on the ocr result and show the visualization.

- CL interface:

```shell
python mmocr/utils/ocr.py demo/demo_kie.jpeg  --det PS_CTW --recog SAR --kie SDMGR --print-result --imshow
```

```{note}

Note: When calling the script from the command line, the script assumes configs are saved in the `configs/` folder. User can customize the directory by specifying the value of `config_dir`.

```

- Python interface:

```python
from mmocr.utils.ocr import MMOCR

# Load models into memory
ocr = MMOCR(det='PS_CTW', recog='SAR', kie='SDMGR')

# Inference
results = ocr.readtext('demo/demo_kie.jpeg', print_result=True, imshow=True)
```

______________________________________________________________________

## API Arguments

The API has an extensive list of arguments that you can use. The following tables are for the python interface.

**MMOCR():**

| Arguments      | Type                  | Default    | Description                                                                                          |
| -------------- | --------------------- | ---------- | ---------------------------------------------------------------------------------------------------- |
| `det`          | see [models](#models) | PANet_IC15 | Text detection algorithm                                                                             |
| `recog`        | see [models](#models) | SAR        | Text recognition algorithm                                                                           |
| `kie` \[1\]    | see [models](#models) | None       | Key information extraction algorithm                                                                 |
| `config_dir`   | str                   | configs/   | Path to the config directory where all the config files are located                                  |
| `det_config`   | str                   | None       | Path to the custom config file of the selected det model                                             |
| `det_ckpt`     | str                   | None       | Path to the custom checkpoint file of the selected det model                                         |
| `recog_config` | str                   | None       | Path to the custom config file of the selected recog model                                           |
| `recog_ckpt`   | str                   | None       | Path to the custom checkpoint file of the selected recog model                                       |
| `kie_config`   | str                   | None       | Path to the custom config file of the selected kie model                                             |
| `kie_ckpt`     | str                   | None       | Path to the custom checkpoint file of the selected kie model                                         |
| `device`       | str                   | None       | Device used for inference, accepting all allowed strings by `torch.device`. E.g., 'cuda:0' or 'cpu'. |

\[1\]: `kie` is only effective when both text detection and recognition models are specified.

```{note}

User can use default pretrained models by specifying `det` and/or `recog`, which is equivalent to specifying their corresponding `*_config` and `*_ckpt`. However, manually specifying `*_config` and `*_ckpt` will always override values set by `det` and/or `recog`. Similar rules also apply to `kie`, `kie_config` and `kie_ckpt`.

```

### readtext()

| Arguments           | Type                    | Default      | Description                                                            |
| ------------------- | ----------------------- | ------------ | ---------------------------------------------------------------------- |
| `img`               | str/list/tuple/np.array | **required** | img, folder path, np array or list/tuple (with img paths or np arrays) |
| `output`            | str                     | None         | Output result visualization - img path or folder path                  |
| `batch_mode`        | bool                    | False        | Whether use batch mode for inference \[1\]                             |
| `det_batch_size`    | int                     | 0            | Batch size for text detection (0 for max size)                         |
| `recog_batch_size`  | int                     | 0            | Batch size for text recognition (0 for max size)                       |
| `single_batch_size` | int                     | 0            | Batch size for only detection or recognition                           |
| `export`            | str                     | None         | Folder where the results of each image are exported                    |
| `export_format`     | str                     | json         | Format of the exported result file(s)                                  |
| `details`           | bool                    | False        | Whether include the text boxes coordinates and confidence values       |
| `imshow`            | bool                    | False        | Whether to show the result visualization on screen                     |
| `print_result`      | bool                    | False        | Whether to show the result for each image                              |
| `merge`             | bool                    | False        | Whether to merge neighboring boxes \[2\]                               |
| `merge_xdist`       | float                   | 20           | The maximum x-axis distance to merge boxes                             |

\[1\]: Make sure that the model is compatible with batch mode.

\[2\]: Only effective when the script is running in det + recog mode.

All arguments are the same for the cli, all you need to do is add 2 hyphens at the beginning of the argument and replace underscores by hyphens.
(*Example:* `det_batch_size` becomes `--det-batch-size`)

For bool type arguments, putting the argument in the command stores it as true.
(*Example:* `python mmocr/utils/ocr.py demo/demo_text_det.jpg --batch_mode --print_result`
means that `batch_mode` and `print_result` are set to `True`)

______________________________________________________________________

## Models

**Text detection:**

| Name          |                                                                        Reference                                                                         | `batch_mode` inference support |
| ------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------: |
| DB_r18        |            [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#real-time-scene-text-detection-with-differentiable-binarization)            |              :x:               |
| DB_r50        |            [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#real-time-scene-text-detection-with-differentiable-binarization)            |              :x:               |
| DBPP_r50      |                                        [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#dbnetpp)                                        |              :x:               |
| DRRG          |                                         [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#drrg)                                          |              :x:               |
| FCE_IC15      |             [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)             |              :x:               |
| FCE_CTW_DCNv2 |             [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#fourier-contour-embedding-for-arbitrary-shaped-text-detection)             |              :x:               |
| MaskRCNN_CTW  |                                      [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#mask-r-cnn)                                       |              :x:               |
| MaskRCNN_IC15 |                                      [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#mask-r-cnn)                                       |              :x:               |
| MaskRCNN_IC17 |                                      [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#mask-r-cnn)                                       |              :x:               |
| PANet_CTW     | [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#efficient-and-accurate-arbitrary-shaped-text-detection-with-pixel-aggregation-network) |       :heavy_check_mark:       |
| PANet_IC15    | [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#efficient-and-accurate-arbitrary-shaped-text-detection-with-pixel-aggregation-network) |       :heavy_check_mark:       |
| PS_CTW        |                                        [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#psenet)                                         |              :x:               |
| PS_IC15       |                                        [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#psenet)                                         |              :x:               |
| Tesseract     |                                                         [link](https://tesseract-ocr.github.io/)                                                         |              :x:               |
| TextSnake     |                                       [link](https://mmocr.readthedocs.io/en/latest/textdet_models.html#textsnake)                                       |       :heavy_check_mark:       |

**Text recognition:**

| Name          |                                                                           Reference                                                                           | `batch_mode` inference support |
| ------------- | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------: |
| ABINet        | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#read-like-humans-autonomous-bidirectional-and-iterative-language-modeling-for-scene-text-recognition) |       :heavy_check_mark:       |
| CRNN          | [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#an-end-to-end-trainable-neural-network-for-image-based-sequence-recognition-and-its-application-to-scene-text-recognition) |              :x:               |
| CRNN_TPS      |                                 [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#crnn-with-tps-based-stn)                                  |       :heavy_check_mark:       |
| MASTER        |                                          [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#master)                                          |       :heavy_check_mark:       |
| NRTR_1/16-1/8 |                                           [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#nrtr)                                           |       :heavy_check_mark:       |
| NRTR_1/8-1/4  |                                           [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#nrtr)                                           |       :heavy_check_mark:       |
| RobustScanner |     [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#robustscanner-dynamically-enhancing-positional-clues-for-robust-text-recognition)     |       :heavy_check_mark:       |
| SAR           |     [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition)     |       :heavy_check_mark:       |
| SAR_CN \*     |     [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#show-attend-and-read-a-simple-and-strong-baseline-for-irregular-text-recognition)     |       :heavy_check_mark:       |
| SATRN         |                                          [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#satrn)                                           |       :heavy_check_mark:       |
| SATRN_sm      |                                          [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#satrn)                                           |       :heavy_check_mark:       |
| SEG           |                                  [link](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#segocr-simple-baseline)                                  |              :x:               |
| Tesseract     |                                                           [link](https://tesseract-ocr.github.io/)                                                            |              :x:               |

```{warning}

SAR_CN is the only model that supports Chinese character recognition and it requires
a Chinese dictionary. Please download the dictionary from [here](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#chinese-dataset) for a successful run.

```

**Key information extraction:**

| Name  |                                                              Reference                                                              | `batch_mode` support |
| ----- | :---------------------------------------------------------------------------------------------------------------------------------: | :------------------: |
| SDMGR | [link](https://mmocr.readthedocs.io/en/latest/kie_models.html#spatial-dual-modality-graph-reasoning-for-key-information-extraction) |  :heavy_check_mark:  |

## Additional info

- To perform det + recog inference (end2end ocr), both the `det` and `recog` arguments must be defined.
- To perform only detection set the `recog` argument to `None`.
- To perform only recognition set the `det` argument to `None`.
- `details` argument only works with end2end ocr.
- `det_batch_size` and `recog_batch_size` arguments define the number of images you want to forward to the model at the same time. For maximum speed, set this to the highest number you can. The max batch size is limited by the model complexity and the GPU VRAM size.
- MMOCR calls Tesseract's API via [`tesserocr`](https://github.com/sirfz/tesserocr)

If you have any suggestions for new features, feel free to open a thread or even PR :)
