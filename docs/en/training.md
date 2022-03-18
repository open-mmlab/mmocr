# Training

## Training on a Single GPU

You can use `tools/train.py` to train a model on a single machine with a CPU and optionally a GPU.

Here is the full usage of the script:

```shell
python tools/train.py ${CONFIG_FILE} [ARGS]
```

:::{note}
By default, MMOCR prefers GPU to CPU. If you want to train a model on CPU, please empty `CUDA_VISIBLE_DEVICES` or set it to -1 to make GPU invisible to the program. Note that CPU training requires **MMCV >= 1.4.4**.

```bash
CUDA_VISIBLE_DEVICES= python tools/train.py ${CONFIG_FILE} [ARGS]
```

:::

| ARGS              | Type                              | Description                                                                                                                                                                                                                                                                                                                                                                            |
| ----------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--work-dir`      | str                               | The target folder to save logs and checkpoints. Defaults to `./work_dirs`.                                                                                                                                                                                                                                                                                                             |
| `--load-from`     | str                               | The checkpoint file to load from.                                                                                                                                                                                                                                                                                                                                                      |
| `--resume-from`   | bool                              | The checkpoint file to resume the training from.                                                                                                                                                                                                                                                                                                                                       |
| `--no-validate`   | bool                              | Disable checkpoint evaluation during training. Defaults to `False`.                                                                                                                                                                                                                                                                                                                    |
| `--gpus`          | int                               | **Deprecated, please use --gpu-id.** Numbers of gpus to use. Only applicable to non-distributed training.                                                                                                                                                                                                                                                                              |
| `--gpu-ids`       | int*N                             | **Deprecated, please use --gpu-id.** A list of GPU ids to use. Only applicable to non-distributed training.                                                                                                                                                                                                                                                                            |
| `--gpu-id`        | int                               | The GPU id to use. Only applicable to non-distributed training.                                                                                                                                                                                                                                                                                                                        |
| `--seed`          | int                               | Random seed.                                                                                                                                                                                                                                                                                                                                                                           |
| `--diff_seed`          | bool                               | Whether or not set different seeds for different ranks.                                                                                                                                                                                                                                                                                                                                                                           |
| `--deterministic` | bool                              | Whether to set deterministic options for CUDNN backend.                                                                                                                                                                                                                                                                                                                                |
| `--cfg-options`   | str                               | Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file. If the value to be overwritten is a list, it should be of the form of either key="[a,b]" or key=a,b. The argument also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]". Note that the quotation marks are necessary and that no white space is allowed. |
| `--launcher`      | 'none', 'pytorch', 'slurm', 'mpi' | Options for job launcher.                                                                                                                                                                                                                                                                                                                                                              |
| `--local_rank`    | int                               | Used for distributed training.                                                                                                                                                                                                                                                                                                                                                         |
| `--mc-config`     | str                               | Memory cache config for image loading speed-up during training.                                                                                                                                                                                                                                                                                                                        |

## Training on Multiple GPUs

MMOCR implements **distributed** training with `MMDistributedDataParallel`. (Please refer to [datasets.md](datasets.md) to prepare your datasets)

```shell
[PORT={PORT}] ./tools/dist_train.sh ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} [PY_ARGS]
```

| Arguments | Type | Description                                                                                                                                                                                                                                      |
| --------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `PORT`    | int  | The master port that will be used by the machine with rank 0. Defaults to 29500. **Note:** If you are launching multiple distrbuted training jobs on a single machine, you need to specify different ports for each job to avoid port conflicts. |
| `CONFIG_FILE`          | str  | The path to config.                                                  |
| `CHECKPOINT_FILE`          | str  | The path to the checkpoint.                                                  |
| `GPU_NUM`          | int  | The number of GPUs to be used per node. Defaults to 8.                                                  |
| `PY_ARGS` | str  | Arguments to be parsed by `tools/train.py`.                                                                                                                                                                                                      |

## Training on Multiple Machines

You can launch a task on multiple machines connected to the same network.

```shell
NNODES=${NNODES} NODE_RANK=${NODE_RANK} PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} ./tools/dist_train.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

| Arguments       | Type | Description                                                                                                 |
| --------------- | ---- | ----------------------------------------------------------------------------------------------------------- |
| `NNODES`          | int  | The number of nodes.
| `NODE_RANK`          | int  | The rank of current node.
| `PORT`    | int  | The master port that will be used by rank 0 node. Defaults to 29500. |
| `MASTER_ADDR`    | int  | The address of rank 0 node. Defaults to "127.0.0.1". |
| `CONFIG_FILE`          | str  | The path to config.                                                  |
| `CHECKPOINT_FILE`          | str  | The path to the checkpoint.                                                  |
| `GPU_NUM`          | int  | The number of GPUs to be used per node. Defaults to 8.                                                  |
| `PY_ARGS`       | str  | Arguments to be parsed by `tools/train.py`.                                                                  |


Say that you want to launch a job on two machines. On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} ./tools/dist_train.sh ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} [PY_ARGS]
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} ./tools/dist_train.sh ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} [PY_ARGS]
```

:::{note}
The speed of the network could be the bottleneck of training.
:::

MMOCR relies on torch.distributed package for distributed training. Find more information at PyTorch’s [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).

## Training with Slurm

If you run MMOCR on a cluster managed with [Slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`.

```shell
[GPUS=${GPUS}] [GPUS_PER_NODE=${GPUS_PER_NODE}] [CPUS_PER_TASK=${CPUS_PER_TASK}] [SRUN_ARGS=${SRUN_ARGS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} [PY_ARGS]
```

| Arguments       | Type | Description                                                                                                 |
| --------------- | ---- | ----------------------------------------------------------------------------------------------------------- |
| `GPUS`          | int  | The number of GPUs to be used by this task. Defaults to 8.                                                  |
| `GPUS_PER_NODE` | int  | The number of GPUs to be allocated per node. Defaults to 8.                                                 |
| `CPUS_PER_TASK` | int  | The number of CPUs to be allocated per task. Defaults to 5.                                                 |
| `SRUN_ARGS`     | str  | Arguments to be parsed by srun. Available options can be found [here](https://slurm.schedmd.com/srun.html). |
| `PY_ARGS`       | str  | Arguments to be parsed by `tools/train.py`.                                                                 |

Here is an example of using 8 GPUs to train a text detection model on the dev partition.

```shell
./tools/slurm_train.sh dev psenet-ic15 configs/textdet/psenet/psenet_r50_fpnf_sbn_1x_icdar2015.py /nfs/xxxx/psenet-ic15
```

### Running Multiple Training Jobs on a Single Machine

If you are launching multiple training jobs on a single machine with Slurm, you may need to modify the port in configs to avoid communication conflicts.

For example, in `config1.py`,

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

## Commonly Used Training Configs

Here we list some configs that are frequently used during training for quick reference.

```python
total_epochs = 1200
data = dict(
    # Note: User can configure general settings of train, val and test dataloader by specifying them here. However, their values can be overridden in dataloader's config.
    samples_per_gpu=8, # Batch size per GPU
    workers_per_gpu=4, # Number of workers to process data for each GPU
    train_dataloader=dict(samples_per_gpu=10, drop_last=True),   # Batch size = 10, workers_per_gpu = 4
    val_dataloader=dict(samples_per_gpu=6, workers_per_gpu=1),  # Batch size = 6, workers_per_gpu = 1
    test_dataloader=dict(workers_per_gpu=16),  # Batch size = 8, workers_per_gpu = 16
    ...
)
# Evaluation
evaluation = dict(interval=1, by_epoch=True)  # Evaluate the model every epoch
# Saving and Logging
checkpoint_config = dict(interval=1)  # Save a checkpoint every epoch
log_config = dict(
    interval=5,  # Print out the model's performance every 5 iterations
    hooks=[
        dict(type='TextLoggerHook')
    ])
# Optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)  # Supports all optimizers in PyTorch and shares the same parameters
optimizer_config = dict(grad_clip=None)  # Parameters for the optimizer hook. See https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py for implementation details
# Learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)
```
