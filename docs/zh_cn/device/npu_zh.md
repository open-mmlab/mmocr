# NPU (华为昇腾)

## 使用方法

首先，请参考[MMCV](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html#npu-mmcv-full) 安装带有 NPU 支持的 MMCV。
使用如下命令，可以利用 4 个 NPU 训练模型（以 CRNN为例）：

```shell
bash tools/dist_train.sh configs/textrecog/crnn/crnn_academic_dataset.py 4
```

或者，使用如下命令，在一个 NPU 上训练模型（以 CRNN为例）：

```shell
python tools/train.py configs/textrecog/crnn/crnn_academic_dataset.py
```

## 经过验证的模型

|   Model    | mean_word_acc_ignore_case | mean_word_acc_ignore_case_symbol | Config                                                            | Download                                                             |
| :--------: | :-----------------------: | :------------------------------: | :---------------------------------------------------------------- | :------------------------------------------------------------------- |
| [CRNN](<>) |           68.4            |               68.7               | [config](https://github.com/open-mmlab/mmocr/blob/0.x/configs/textrecog/crnn/crnn_academic_dataset.py) | [log](https://download.openmmlab.com/mmocr/textrecog/crnn/crnn_20230406_103202.log.json) |

**注意:**

- 如果没有特别标记，NPU 上的结果与使用 FP32 的 GPU 上的结果结果相同。

**以上所有模型权重及训练日志均由华为昇腾团队提供**
