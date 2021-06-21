# Getting Started

This page provides basic tutorials on the usage of MMOCR.
For the installation instructions, please see [install.md](install.md).


## Inference with Pretrained Models

We provide testing scripts to evaluate a full dataset, as well as some task-specific image demos.

### Test a Single Image

You can use the following command to test a single image with one GPU.

```shell
python demo/image_demo.py ${TEST_IMG} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SAVE_PATH} [--imshow] [--device ${GPU_ID}]
```

If `--imshow` is specified, the demo will also show the image with OpenCV. For example:

```shell
python demo/image_demo.py demo/demo_text_det.jpg configs/xxx.py xxx.pth demo/demo_text_det_pred.jpg
```

The predicted result will be saved as `demo/demo_text_det_pred.jpg`.

To end-to-end test a single image with both text detection and recognition,

```shell
python demo/ocr_image_demo.py demo/demo_text_det.jpg demo/output.jpg
```

The predicted result will be saved as `demo/output.jpg`.

### Test Multiple Images

```shell
# for text detection
./tools/det_test_imgs.py ${IMG_ROOT_PATH} ${IMG_LIST} ${CONFIG_FILE} ${CHECKPOINT_FILE} --out-dir ${RESULTS_DIR}

# for text recognition
./tools/recog_test_imgs.py ${IMG_ROOT_PATH} ${IMG_LIST} ${CONFIG_FILE} ${CHECKPOINT_FILE} --out-dir ${RESULTS_DIR}
```
It will save both the prediction results and visualized images to `${RESULTS_DIR}`

### Test a Dataset

MMOCR implements **distributed** testing with `MMDistributedDataParallel`. (Please refer to [datasets.md](datasets.md) to prepare your datasets)

#### Test with Single/Multiple GPUs

You can use the following command to test a dataset with single/multiple GPUs.

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--eval ${EVAL_METRIC}]
```
For example,

```shell
./tools/dist_test.sh configs/example_config.py work_dirs/example_exp/example_model_20200202.pth 1 --eval hmean-iou
```
##### Optional Arguments

- `--eval`: Specify the evaluation metric. For text detection, the metric should be either 'hmean-ic13' or 'hmean-iou'. For text recognition, the metric should be 'acc'.

#### Test with Slurm

If you run MMOCR on a cluster managed with [Slurm](https://slurm.schedmd.com/), you can use the script `slurm_test.sh`.

```shell
[GPUS=${GPUS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--eval ${EVAL_METRIC}]
```
Here is an example of using 8 GPUs to test an example model on the 'dev' partition with job name 'test_job'.

```shell
GPUS=8 ./tools/slurm_test.sh dev test_job configs/example_config.py work_dirs/example_exp/example_model_20200202.pth --eval hmean-iou
```

You can check [slurm_test.sh](https://github.com/open-mmlab/mmocr/blob/master/tools/slurm_test.sh) for full arguments and environment variables.


##### Optional Arguments

- `--eval`: Specify the evaluation metric. For text detection, the metric should be either 'hmean-ic13' or 'hmean-iou'. For text recognition, the metric should be 'acc'.


## Train a Model

MMOCR implements **distributed** training with `MMDistributedDataParallel`. (Please refer to [datasets.md](datasets.md) to prepare your datasets)

All outputs (log files and checkpoints) will be saved to a working directory specified by `work_dir` in the config file.

By default, we evaluate the model on the validation set after several iterations. You can change the evaluation interval by adding the interval argument in the training config as follows:
```python
evaluation = dict(interval=1, by_epoch=True)  # This evaluates the model per epoch.
```


### Train with Single/Multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} [optional arguments]
```

Optional Arguments:

- `--no-validate` (**not suggested**): By default, the codebase will perform evaluation at every k-th iteration during training. To disable this behavior, use `--no-validate`.

#### Train with Toy Dataset.
We provide a toy dataset under `tests/data`, and you can train a toy model directly, before the academic dataset is prepared.

For example, train a text recognition task with `seg` method and toy dataset,
```
./tools/dist_train.sh configs/textrecog/seg/seg_r31_1by16_fpnocr_toy_dataset.py work_dirs/seg 1
```

And train a text recognition task with `sar` method and toy dataset,
```
./tools/dist_train.sh configs/textrecog/sar/sar_r31_parallel_decoder_toy_dataset.py work_dirs/sar 1
```

### Train with Slurm

If you run MMOCR on a cluster managed with [Slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`.

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

Here is an example of using 8 GPUs to train a text detection model on the dev partition.

```shell
GPUS=8 ./tools/slurm_train.sh dev psenet-ic15 configs/textdet/psenet/psenet_r50_fpnf_sbn_1x_icdar2015.py /nfs/xxxx/psenet-ic15
```

You can check [slurm_train.sh](https://github.com/open-mmlab/mmocr/blob/master/tools/slurm_train.sh) for full arguments and environment variables.

### Launch Multiple Jobs on a Single Machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflicts.

If you use `dist_train.sh` to launch training jobs, you can set the ports in the command shell.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

If you launch training jobs with Slurm, you need to modify the config files to set different communication ports.

In `config1.py`,
```python
dist_params = dict(backend='nccl', port=29500)
```

In `config2.py`,
```python
dist_params = dict(backend='nccl', port=29501)
```

Then you can launch two jobs with `config1.py` ang `config2.py`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
```


## Useful Tools

We provide numerous useful tools under `mmocr/tools` directory.

### Publish a Model

Before you upload a model to AWS, you may want to
(1) convert the model weights to CPU tensors, (2) delete the optimizer states and
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/psenet/latest.pth psenet_r50_fpnf_sbn_1x_20190801.pth
```

The final output filename will be `psenet_r50_fpnf_sbn_1x_20190801-{hash id}.pth`.

## Customized Settings

### Flexible Dataset
To support the tasks of `text detection`, `text recognition` and `key information extraction`, we have designed a new type of dataset which consists of `loader` and `parser` to load and parse different types of annotation files.
- **loader**: Load the annotation file. There are two types of loader, `HardDiskLoader` and `LmdbLoader`
  - `HardDiskLoader`: Load `txt` format annotation file from hard disk to memory.
  - `LmdbLoader`: Load `lmdb` format annotation file with lmdb backend, which is very useful for **extremely large** annotation files to avoid out-of-memory problem when ten or more GPUs are used, since each GPU will start multiple processes to load annotation file to memory.
- **parser**: Parse the annotation file line-by-line and return with `dict` format. There are two types of parser, `LineStrParser` and `LineJsonParser`.
  - `LineStrParser`: Parse one line in ann file while treating it as a string and separating it to several parts by a `separator`. It can be used on tasks with simple annotation files such as text recognition where each line of the annotation files contains the `filename` and `label` attribute only.
  - `LineJsonParser`: Parse one line in ann file while treating it as a json-string and using `json.loads` to convert it to `dict`. It can be used on tasks with complex annotation files such as text detection where each line of the annotation files contains multiple attributes (e.g. `filename`, `height`, `width`, `box`, `segmentation`, `iscrowd`, `category_id`, etc.).

Here we show some examples of using different combination of `loader` and `parser`.

#### Text Recognition Task

##### OCRDataset

<small>*Dataset for encoder-decoder based recognizer*</small>

```python
dataset_type = 'OCRDataset'
img_prefix = 'tests/data/ocr_toy_dataset/imgs'
train_anno_file = 'tests/data/ocr_toy_dataset/label.txt'
train = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=10,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)
```
You can check the content of the annotation file in `tests/data/ocr_toy_dataset/label.txt`.
The combination of `HardDiskLoader` and `LineStrParser` will return a dict for each file by calling `__getitem__`: `{'filename': '1223731.jpg', 'text': 'GRAND'}`.

**Optional Arguments:**

- `repeat`: The number of repeated lines in the annotation files. For example, if there are `10` lines in the annotation file, setting `repeat=10` will generate a corresponding annotation file with size `100`.

If the annotation file is extreme large, you can convert it from txt format to lmdb format with the following command:
```python
python tools/data_converter/txt2lmdb.py -i ann_file.txt -o ann_file.lmdb
```

After that, you can use `LmdbLoader` in dataset like below.
```python
img_prefix = 'tests/data/ocr_toy_dataset/imgs'
train_anno_file = 'tests/data/ocr_toy_dataset/label.lmdb'
train = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file,
    loader=dict(
        type='LmdbLoader',
        repeat=10,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)
```

##### OCRSegDataset

<small>*Dataset for segmentation-based recognizer*</small>

```python
prefix = 'tests/data/ocr_char_ann_toy_dataset/'
train = dict(
    type='OCRSegDataset',
    img_prefix=prefix + 'imgs',
    ann_file=prefix + 'instances_train.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=10,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'annotations', 'text'])),
    pipeline=train_pipeline,
    test_mode=True)
```
You can check the content of the annotation file in `tests/data/ocr_char_ann_toy_dataset/instances_train.txt`.
The combination of `HardDiskLoader` and `LineJsonParser` will return a dict for each file by calling `__getitem__` each time:
```python
{"file_name": "resort_88_101_1.png", "annotations": [{"char_text": "F", "char_box": [11.0, 0.0, 22.0, 0.0, 12.0, 12.0, 0.0, 12.0]}, {"char_text": "r", "char_box": [23.0, 2.0, 31.0, 1.0, 24.0, 11.0, 16.0, 11.0]}, {"char_text": "o", "char_box": [33.0, 2.0, 43.0, 2.0, 36.0, 12.0, 25.0, 12.0]}, {"char_text": "m", "char_box": [46.0, 2.0, 61.0, 2.0, 53.0, 12.0, 39.0, 12.0]}, {"char_text": ":", "char_box": [61.0, 2.0, 69.0, 2.0, 63.0, 12.0, 55.0, 12.0]}], "text": "From:"}
```

#### Text Detection Task

##### TextDetDataset

<small>*Dataset with annotation file in line-json txt format*</small>

```python
dataset_type = 'TextDetDataset'
img_prefix = 'tests/data/toy_dataset/imgs'
test_anno_file = 'tests/data/toy_dataset/instances_test.txt'
test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=test_anno_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=4,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=test_pipeline,
    test_mode=True)
```
The results are generated in the same way as the segmentation-based text recognition task above.
You can check the content of the annotation file in `tests/data/toy_dataset/instances_test.txt`.
The combination of `HardDiskLoader` and `LineJsonParser` will return a dict for each file by calling `__getitem__`:
```python
{"file_name": "test/img_10.jpg", "height": 720, "width": 1280, "annotations": [{"iscrowd": 1, "category_id": 1, "bbox": [260.0, 138.0, 24.0, 20.0], "segmentation": [[261, 138, 284, 140, 279, 158, 260, 158]]}, {"iscrowd": 0, "category_id": 1, "bbox": [288.0, 138.0, 129.0, 23.0], "segmentation": [[288, 138, 417, 140, 416, 161, 290, 157]]}, {"iscrowd": 0, "category_id": 1, "bbox": [743.0, 145.0, 37.0, 18.0], "segmentation": [[743, 145, 779, 146, 780, 163, 746, 163]]}, {"iscrowd": 0, "category_id": 1, "bbox": [783.0, 129.0, 50.0, 26.0], "segmentation": [[783, 129, 831, 132, 833, 155, 785, 153]]}, {"iscrowd": 1, "category_id": 1, "bbox": [831.0, 133.0, 43.0, 23.0], "segmentation": [[831, 133, 870, 135, 874, 156, 835, 155]]}, {"iscrowd": 1, "category_id": 1, "bbox": [159.0, 204.0, 72.0, 15.0], "segmentation": [[159, 205, 230, 204, 231, 218, 159, 219]]}, {"iscrowd": 1, "category_id": 1, "bbox": [785.0, 158.0, 75.0, 21.0], "segmentation": [[785, 158, 856, 158, 860, 178, 787, 179]]}, {"iscrowd": 1, "category_id": 1, "bbox": [1011.0, 157.0, 68.0, 16.0], "segmentation": [[1011, 157, 1079, 160, 1076, 173, 1011, 170]]}]}
```


##### IcdarDataset

<small>*Dataset with annotation file in coco-like json format*</small>

For text detection, you can also use an annotation file in a COCO format that is defined in [mmdet](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py):
```python
dataset_type = 'IcdarDataset'
prefix = 'tests/data/toy_dataset/'
test=dict(
        type=dataset_type,
        ann_file=prefix + 'instances_test.json',
        img_prefix=prefix + 'imgs',
        pipeline=test_pipeline)
```
You can check the content of the annotation file in `tests/data/toy_dataset/instances_test.json`
- The icdar2015/2017 annotations have to be converted into the COCO format using `tools/data_converter/icdar_converter.py`:

  ```shell
  python tools/data_converter/icdar_converter.py ${src_root_path} -o ${out_path} -d ${data_type} --split-list training validation test
  ```

- The ctw1500 annotations have to be converted into the COCO format using `tools/data_converter/ctw1500_converter.py`:

  ```shell
  python tools/data_converter/ctw1500_converter.py ${src_root_path} -o ${out_path} --split-list training test
  ```

#### UniformConcatDataset

To use the same pipeline for multiple datasets, as well as support extra parameters such as `samples_per_gpu` in `cfg.data.test` to activate `batch inference`, we design `UniformConcatDataset`. See [config](/configs/textrecog/sar/sar_r31_parallel_decoder_toy_dataset.py) for an example.

If you want to use `batch inference` during testing, please update the config file refering to line108 - line117 in [config](/configs/textrecog/sar/sar_r31_parallel_decoder_toy_dataset.py)
