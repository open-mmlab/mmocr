# Testing

We introduce the way to test pretrained models on datasets here.

## Testing with Single GPU

You can use `tools/test.py` to perform single GPU inference. For example, to evaluate DBNet on IC15: (You can download pretrained models from [Model Zoo](modelzoo.md)):

```shell
./tools/dist_test.sh configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth --eval hmean-iou
```

And here is the full usage of the script:

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```


| ARGS      | Type                  |  Description                                                 |
| -------------- | --------------------- |  ----------------------------------------------------------- |
| `--out`          | str                   |  Output result file in pickle format. |
| `--fuse-conv-bn`   | bool                   |  Path to the custom config of the selected det model.        |
| `--format-only`        | bool |  Format the output results without performing evaluation. It is useful when you want to format the results to a specific format and submit them to the test server.|
| `--eval` | 'hmean-ic13', 'hmean-iou', 'acc' |  The evaluation metrics, which depends on the task. For text detection, the metric should be either 'hmean-ic13' or 'hmean-iou'. For text recognition, the metric should be 'acc'. |
| `--show`       | bool                   |  Whether to show results. |
| `--show-dir`       | str                   |  Directory where the output images will be saved. |
| `--show-score-thr`      | float                   |  Score threshold (default: 0.3). |
| `--gpu-collect`       | bool                   |  Whether to use gpu to collect results. |
| `--tmpdir`       | str                   |  The tmp directory used for collecting results from multiple workers, available when gpu-collect is not specified.                |
| `--cfg-options`       | str                   |          Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file. If the value to be overwritten is a list, it should be of the form of either key="[a,b]" or key=a,b. The argument also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]". Note that the quotation marks are necessary and that no white space is allowed.|
| `--eval-options`       | str                   |Custom options for evaluation, the key-value pair in xxx=yyy format will be kwargs for dataset.evaluate() function.|
| `--launcher`       | 'none', 'pytorch', 'slurm', 'mpi' |  Options for job launcher. |


## Testing with Multiple GPUs

MMOCR implements **distributed** testing with `MMDistributedDataParallel`.

You can use the following command to test a dataset with multiple GPUs.

```shell
[PORT={PORT}] ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```


| Arguments      | Type                  |  Description                                                 |
| -------------- | --------------------- |  ----------------------------------------------------------- |
| `PORT`          | int                   |  The master port that will be used by the machine with rank 0. Defaults to 29500. |
| `PY_ARGS`   | str                   |  Arguments to be parsed by `tools/test.py`.         |


For example,

```shell
./tools/dist_test.sh configs/example_config.py work_dirs/example_exp/example_model_20200202.pth 1 --eval hmean-iou
```

## Testing with Slurm

If you run MMOCR on a cluster managed with [Slurm](https://slurm.schedmd.com/), you can use the script `tools/slurm_test.sh`.


```shell
[GPUS=${GPUS}] [GPUS_PER_NODE=${GPUS_PER_NODE}] [SRUN_ARGS=${SRUN_ARGS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE} [PY_ARGS]
```

| Arguments      | Type                  |  Description                                                 |
| -------------- | --------------------- |  ----------------------------------------------------------- |
| `GPUS`          | int                   |  The number of GPUs to be used by this task. Defaults to 8. |
| `GPUS_PER_NODE`   | int                   |  The number of GPUs to be allocated per node. Defaults to 8. |
| `SRUN_ARGS`        | str                   |  Arguments to be parsed by srun. Available options can be found [here](https://slurm.schedmd.com/srun.html). |
| `PY_ARGS`   | str                   |  Arguments to be parsed by `tools/test.py`.         |


Here is an example of using 8 GPUs to test an example model on the 'dev' partition with job name 'test_job'.

```shell
GPUS=8 ./tools/slurm_test.sh dev test_job configs/example_config.py work_dirs/example_exp/example_model_20200202.pth --eval hmean-iou
```

## Batch Testing

By default, MMOCR tests the model image by image. For faster inference, you may change `data.val_dataloader.samples_per_gpu` and `data.test_dataloader.samples_per_gpu` in the config. For example,

```
data = dict(
    ...
    val_dataloader=dict(samples_per_gpu=16),
    test_dataloader=dict(samples_per_gpu=16),
    ...
)
```
will test the model with 16 images in a batch.

:::{warning}
Batch testing may incur performance decrease of the model due to the different behavior of the data preprocessing pipeline.
:::
