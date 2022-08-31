# Quick Run

## Inference

In addition to using our well-provided pre-trained models, you can also train models on your own datasets. In the next section, we will take you through the basic functions of MMOCR by training DBNet on the mini [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) dataset as an example.

The next sections assume that you are using the [editorial approach to install](install.md) the MMOCR codebase.

## Prepare a Dataset

Since the variety of OCR dataset formats are not conducive to either switching or joint training of multiple datasets, MMOCR proposes a uniform [data format](../user_guides/dataset_prepare.md), and provides conversion scripts and [tutorials](../user_guides/dataset_prepare.md) for all commonly used OCR datasets. Usually, to use those datasets in MMOCR, you just need to follow the steps to get them ready for use.

```{note}
But here, efficiency means everything.
```

Here, we have prepared a lite version of ICDAR 2015 dataset for demonstration purposes. Download our pre-prepared [zip](https://download.openmmlab.com/mmocr/data/icdar2015/mini_icdar2015.tar.gz) and extract it to the `data/det/` directory under mmocr to get our prepared image and annotation file.

```Bash
wget https://download.openmmlab.com/mmocr/data/icdar2015/mini_icdar2015.tar.gz
mkdir -p data/det/
tar xzvf mini_icdar2015.tar.gz -C data/det/
```

## Modify the Config

Once the dataset is prepared, we will then specify the location of the training set and the training parameters by modifying the config file.

In this example, we will train a DBNet using resnet18 as its backbone. Since MMOCR already has a config file for the full ICDAR 2015 dataset (`configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py`), we just need to make some modifications on top of it.

We first need to modify the path to the dataset. In this config, most of the key config files are imported in `_base_`, such as the database configuration from `configs/_base_/det_datasets/icdar2015.py`. Open that file and replace the path pointed to by `ic15_det_data_root` in the first line with:

```Python
ic15_det_data_root = 'data/det/mini_icdar2015'
```

Also, because of the reduced dataset size, we have to reduce the number of training epochs to 400 accordingly, shorten the validation interval as well as the weight storage interval to 10 rounds, and drop the learning rate decay strategy. The following lines of configuration can be directly put into `configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py` to take effect.

```Python
# Save checkpoints every 10 epochs
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10), )
# Set the maximum number of epochs to 400, and validate the model every 10 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400, val_interval=10)
# Fix learning rate as a constant
param_scheduler = [dict(type='ConstantLR', factor=1.0),]
```

Here, we have rewritten the corresponding parameters in the base configuration directly through the [inheritance](https://mmengine.readthedocs.io/en/latest/tutorials/config.html) mechanism of the configuration. The original fields are distributed in `configs/_base_/schedules/schedule_sgd_1200e.py` and `configs/_base_/textdet_default_runtime.py`. You may check them out if interested.

```{tip}
For a more detailed description of config, please refer to [here](../user_guides/config.md).
```

## Browse the Dataset

Before we start the training, we can also visualize the image processed by training-time [data transforms](../basic_concepts/transforms.md). It's quite simple: pass the config file we need to visualize into the [browse_dataset.py](/tools/analysis_tools/browse_dataset.py) script.

```Bash
python tools/analysis_tools/browse_dataset.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py
```

The transformed images and annotations will be displayed one by one in a pop-up window.

<center class="half">
    <img src="https://user-images.githubusercontent.com/24622904/187611542-01e9aa94-fc12-4756-964b-a0e472522a3a.jpg" width="250"/><img src="https://user-images.githubusercontent.com/24622904/187611555-3f5ea616-863d-4538-884f-bccbebc2f7e7.jpg" width="250"/><img src="https://user-images.githubusercontent.com/24622904/187611581-88be3970-fbfe-4f62-8cdf-7a8a7786af29.jpg" width="250"/>
</center>

For details on the parameters and usage of this script, please refer to [here](../user_guides/useful_tools.md).

```{tip}
In addition to satisfying our curiosity, visualization can also help us check the parts that may affect the model's performance before training, such as problems in configs, datasets and data transforms.
```

## Training

Start the training by running the following command:

```Bash
python tools/train.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py
```

Depending on the system environment, MMOCR will automatically use the best device for training. If a GPU is available, a single GPU training will be started by default. When you start to see the output of the losses, you have successfully started the training.

```Bash
2022/08/22 18:42:22 - mmengine - INFO - Epoch(train) [1][5/7]  lr: 7.0000e-03  memory: 7730  data_time: 0.4496  loss_prob: 14.6061  loss_thr: 2.2904  loss_db: 0.9879  loss: 17.8843  time: 1.8666
2022/08/22 18:42:24 - mmengine - INFO - Exp name: dbnet_resnet18_fpnc_1200e_icdar2015
2022/08/22 18:42:28 - mmengine - INFO - Epoch(train) [2][5/7]  lr: 7.0000e-03  memory: 6695  data_time: 0.2052  loss_prob: 6.7840  loss_thr: 1.4114  loss_db: 0.9855  loss: 9.1809  time: 0.7506
2022/08/22 18:42:29 - mmengine - INFO - Exp name: dbnet_resnet18_fpnc_1200e_icdar2015
2022/08/22 18:42:33 - mmengine - INFO - Epoch(train) [3][5/7]  lr: 7.0000e-03  memory: 6690  data_time: 0.2101  loss_prob: 3.0700  loss_thr: 1.1800  loss_db: 0.9967  loss: 5.2468  time: 0.6244
2022/08/22 18:42:33 - mmengine - INFO - Exp name: dbnet_resnet18_fpnc_1200e_icdar2015
```

Without extra configurations, model weights will be saved to `work_dirs/dbnet_resnet18_fpnc_1200e_icdar2015/`, while the logs will be stored in `work_dirs/dbnet_resnet18_fpnc_1200e_icdar2015/TIMESTAMP/`. Next, we just need to wait with some patience for training to finish.

```{tip}
For advanced usage of training, such as CPU training, multi-GPU training, and cluster training, please refer to [Training and Testing](../user_guides/train_test.md).
```

## Testing

After 400 epochs, we observe that DBNet performs best in the last epoch, with `hmean` reaching 60.86:

```Bash
08/22 19:24:52 - mmengine - INFO - Epoch(val) [400][100/100]  icdar/precision: 0.7285  icdar/recall: 0.5226  icdar/hmean: 0.6086
```

```{note}
It may not have been trained to be optimal, but it is sufficient for a demo.
```

However, this value only reflects the performance of DBNet on the mini ICDAR 2015 dataset. For a comprehensive evaluation, we also need to see how it performs on out-of-distribution datasets. For example, `tests/data/det_toy_dataset` is a very small real dataset that we can use to verify the actual performance of DBNet.

Before testing, we also need to make some changes to the location of the dataset. Open `configs/_base_/det_datasets/icdar2015.py` and change `data_root` of `ic15_det_test` to `tests/data/det_toy_dataset`:

```Python
# ...
ic15_det_test = dict(
    type='OCRDataset',
    data_root='tests/data/det_toy_dataset',
    #  ...
    )
```

Start testing:

```Bash
python tools/test.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py work_dirs/dbnet_resnet18_fpnc_1200e_icdar2015/epoch_400.pth
```

And get the outputs:

```Bash
08/21 21:45:59 - mmengine - INFO - Epoch(test) [5/10]    memory: 8562
08/21 21:45:59 - mmengine - INFO - Epoch(test) [10/10]    eta: 0:00:00  time: 0.4893  data_time: 0.0191  memory: 283
08/21 21:45:59 - mmengine - INFO - Evaluating hmean-iou...
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.30, recall: 0.6190, precision: 0.4815, hmean: 0.5417
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.40, recall: 0.6190, precision: 0.5909, hmean: 0.6047
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.50, recall: 0.6190, precision: 0.6842, hmean: 0.6500
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.60, recall: 0.6190, precision: 0.7222, hmean: 0.6667
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.70, recall: 0.3810, precision: 0.8889, hmean: 0.5333
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.80, recall: 0.0000, precision: 0.0000, hmean: 0.0000
08/21 21:45:59 - mmengine - INFO - prediction score threshold: 0.90, recall: 0.0000, precision: 0.0000, hmean: 0.0000
08/21 21:45:59 - mmengine - INFO - Epoch(test) [10/10]  icdar/precision: 0.7222  icdar/recall: 0.6190  icdar/hmean: 0.6667
```

The model achieves an hmean of 0.6667 on this dataset.

```{tip}
For advanced usage of testing, such as CPU testing, multi-GPU testing, and cluster testing, please refer to [Training and Testing] (../user_guides/train_test.md).
```

## Visualize the Outputs

We can also visualize its prediction output in `test.py`. You can open a pop-up visualization window with the `show` parameter; and can also specify the directory where the prediction result images are exported with the `show-dir` parameter.

```Bash
python tools/test.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py work_dirs/dbnet_r18_fpnc_1200e_icdar2015/epoch_400.pth --show-dir imgs/
```

The true labels and predicted values are displayed in a tiled fashion in the visualization results. The green boxes in the left panel indicate the true labels and the red boxes in the right panel indicate the predicted values.

<div align="center">
    <img src="https://user-images.githubusercontent.com/22607038/187423562-6a85e209-4b12-46ee-8a41-5c67b1ba83f9.png"/><br>
</div>

```{tip}
For a description of more visualization features, see [here](../user_guides/visualization.md).
```
