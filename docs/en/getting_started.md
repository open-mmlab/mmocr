# Getting Started

In this guide, we will show you some useful commands and familiarize you with MMOCR.

## Installation

Requirements:

- MMEngine
- MMCV
- MMDetection

```bash
# Create conda environment
conda create -n mmocr python=3.8 -y
conda activate mmocr
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install requirements
# TODO: Update after release
# Engine
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine && pip install -e . && cd ../
# MMCV
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv && git checkout dev-2.x
# Compiling on cluster
MMCV_WITH_OPS=1 srun -p $PARTITION -n1 --gres=gpu:1 pip install -e .
# Compiling on local machine
MMCV_WITH_OPS=1 pip install -e .
# MMDet
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && git checkout dev-3.x
pip install -e .

# Install MMOCR
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr && git checkout dev-1.x
pip install -e .
```

## Dataset Preparation

MMOCR supports numerous datasets which are classified by the type of their corresponding tasks. You may find their preparation steps in these sections: [Detection Datasets](https://mmocr.readthedocs.io/en/latest/datasets/det.html), [Recognition Datasets](https://mmocr.readthedocs.io/en/latest/datasets/recog.html), and [KIE Datasets](https://mmocr.readthedocs.io/en/latest/datasets/kie.html). For a quick start, we also prepared a toy dataset and its corresponding configs, which can be found under `tests/data`.

## Training

### Training with Toy Dataset

Training a text recognizer CRNN on toy dataset.

```bash
python tools/train.py configs/textrecog/crnn/crnn_toy_dataset.py --work-dir crnn
```

### Training with Academic Dataset

Once you have prepared the required academic dataset following [our instructions](https://mmocr.readthedocs.io/en/latest/datasets/det.html), the only last thing to check is if the model’s config points MMOCR to the correct dataset path. Suppose we want to train DBNet on ICDAR 2015; you will need to check the root path in the dataset config (`configs/_base_/det_datasets/icdar2015.py`) correctly points to your local dir. If you are unfamiliar with the config structure in MMOCR, don't worry; please refer to our [config tutorial](<>).

```python
data_root = 'data/det/icdar2015' # check if your dataset is linked to this path

train_anno_path = 'instances_training.json'
test_anno_path = 'instances_test.json'

train_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file=train_anno_path,
    data_prefix=dict(img_path='imgs/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

test_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file=test_anno_path,
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=None)

train_list = [train_dataset]
test_list = [test_dataset]
```

Then you can start training with the command:

```bash
python tools/train.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py --work-dir dbnet
```

## Testing

Suppose now you have finished the training of DBNet and the latest model has been saved in `dbnet/latest.pth`. You can evaluate its performance with the following command:

```shell
python tools/test.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py dbnet/latest.pth
```

To dump the prediction results, you can add `--save-preds`, which will automatically save the predicted bounding boxes/recognition results in `.pkl` format.

## Useful Tools

MMOCR provides useful tools to help users visualize, analyze or evaluate while developing new OCR models.

### Browsing your Datasets

You can use the `tools/analysis_tools/browse_dataset.py` visualization script to browse your customized datasets and training pipelines. For example, using the following command to view the data transformed by a training pipeline used for training DBNet.

```bash
python tools/analysis_tools/browse_dataset.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py --output-dir ./vis_dbnet_ic15
```

### Offline Evaluation

Suppose that you have dumped the prediction results of the DBNet, then you can use the offline evaluation tools to get the performance any time later without re-running the testing script.

```bash
python tools/analysis_tools/offline_eval.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py path/to/results/dbnet_predictions.pkl
```
