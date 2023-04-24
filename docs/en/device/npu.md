# NPU (HUAWEI Ascend)

## Usage

Please refer to the [building documentation of MMCV](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) to install MMCV on NPU devices.

Here we use 4 NPUs on your computer to train the model with the following command:

```shell
bash tools/dist_train.sh configs/textrecog/crnn/crnn_academic_dataset.py 4
```

Also, you can use only one NPU to train the model with the following command:

```shell
python tools/train.py configs/textrecog/crnn/crnn_academic_dataset.py
```

## Models Results

|   Model    | mean_word_acc_ignore_case | mean_word_acc_ignore_case_symbol | Config                                                            | Download                                                             |
| :--------: | :-----------------------: | :------------------------------: | :---------------------------------------------------------------- | :------------------------------------------------------------------- |
| [CRNN](<>) |           68.4            |               68.7               | [config](https://github.com/open-mmlab/mmocr/blob/0.x/configs/textrecog/crnn/crnn_academic_dataset.py) | [log](https://download.openmmlab.com/mmocr/textrecog/crnn/crnn_20230406_103202.log.json) |

**Notes:**

- If not specially marked, the results on NPU with amp are the basically same as those on the GPU with FP32.

**All above models are provided by Huawei Ascend group.**
