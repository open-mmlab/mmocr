# Training and Testing

To meet diverse requirements, MMOCR supports training and testing models on various devices, including PCs, work stations, computation clusters, etc.

## Single GPU Training and Testing

### Training

`tools/train.py` provides the basic training service. MMOCR recommends using GPUs for model training and testing, but it still enables CPU-Only training and testing. For example, the following commands demonstrate how to train a DBNet model using a single GPU or CPU.

```bash
# Train the specified MMOCR model by calling tools/train.py
CUDA_VISIBLE_DEVICES= python tools/train.py ${CONFIG_FILE} [PY_ARGS]

# Training
# Example 1: Training DBNet with CPU
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py

# Example 2: Specify to train DBNet with gpu:0, specify the working directory as dbnet/, and turn on mixed precision (amp) training
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py --work-dir dbnet/ --amp
```

```{note}
If multiple GPUs are available, you can specify a certain GPU, e.g. the third one, by setting CUDA_VISIBLE_DEVICES=3.
```

The following table lists all the arguments supported by `train.py`. Args without the `--` prefix are mandatory, while others are optional.

| ARGS            | Type | Description                                                                 |
| --------------- | ---- | --------------------------------------------------------------------------- |
| config          | str  | (required)Path to config.                                                   |
| --work-dir      | str  | Specify the working directory for the training logs and models checkpoints. |
| --resume        | bool | Whether to resume training from the latest checkpoint.                      |
| --amp           | bool | Whether to use automatic mixture precision for training.                    |
| --auto-scale-lr | bool | Whether to use automatic learning rate scaling.                             |
| --cfg-options   | str  | Override some settings in the configs. [Example](<>)                        |
| --launcher      | str  | Option for launcher，\['none', 'pytorch', 'slurm', 'mpi'\].                 |
| --local_rank    | int  | Rank of local machine，used for distributed training，defaults to 0。       |

### Test

`tools/test.py` provides the basic testing service, which is used in a similar way to the training script. For example, the following command demonstrates test a DBNet model on a single GPU or CPU.

```bash
# Test a pretrained MMOCR model by calling tools/test.py
CUDA_VISIBLE_DEVICES= python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [PY_ARGS]

# Test
# Example 1: Testing DBNet with CPU
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth

# Example 2: Testing DBNet on gpu:0
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth
```

The following table lists all the arguments supported by `test.py`. Args without the `--` prefix are mandatory, while others are optional.

| ARGS          | Type  | Description                                                          |
| ------------- | ----- | -------------------------------------------------------------------- |
| config        | str   | (required)Path to config.                                            |
| checkpoint    | str   | (required)The model to be tested.                                    |
| --work-dir    | str   | Specify the working directory for the logs.                          |
| --save-preds  | bool  | Whether to save the predictions to a pkl file.                       |
| --show        | bool  | Whether to visualize the predictions.                                |
| --show-dir    | str   | Path to save the visualization results.                              |
| --wait-time   | float | Interval of visualization (s), defaults to 2.                        |
| --cfg-options | str   | Override some settings in the configs. [Example](<>)                 |
| --launcher    | str   | Option for launcher，\['none', 'pytorch', 'slurm', 'mpi'\].          |
| --local_rank  | int   | Rank of local machine，used for distributed training，defaults to 0. |

## Training and Testing with Multiple GPUs

For large models, distributed training or testing significantly improves the efficiency. For this purpose, MMOCR provides distributed scripts `tools/dist_train.sh` and `tools/dist_test.sh` implemented based on [MMDistributedDataParallel](mmengine.model.wrappers.MMDistributedDataParallel).

```bash
# Training
NNODES=${NNODES} NODE_RANK=${NODE_RANK} PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]

# Testing
NNODES=${NNODES} NODE_RANK=${NODE_RANK} PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

The following table lists the arguments supported by `dist_*.sh`.

| ARGS            | Type | Description                                                                                   |
| --------------- | ---- | --------------------------------------------------------------------------------------------- |
| NNODES          | int  | The number of nodes. Defaults to 1.                                                           |
| NODE_RANK       | int  | The rank of current node. Defaults to 0.                                                      |
| PORT            | int  | The master port that will be used by rank 0 node, ranging from 0 to 65535. Defaults to 29500. |
| MASTER_ADDR     | str  | The address of rank 0 node. Defaults to "127.0.0.1".                                          |
| CONFIG_FILE     | str  | (required)The path to config.                                                                 |
| CHECKPOINT_FILE | str  | (required，only used in dist_test.sh)The path to checkpoint to be tested.                     |
| GPU_NUM         | int  | (required)The number of GPUs to be used per node.                                             |
| \[PY_ARGS\]     | str  | Arguments to be parsed by tools/train.py and tools/test.py.                                   |

These two scripts enable training and testing on **single-machine multi-GPU** or **multi-machine multi-GPU**. See the following example for usage.

### Single-machine Multi-GPU

The following commands demonstrate how to train and test with a specified number of GPUs on a **single machine** with multiple GPUs.

1. **Training**

   Training DBNet using 4 GPUs on a single machine.

   ```bash
   tools/dist_train.sh configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py 4
   ```

2. **Testing**

   Testing DBNet using 4 GPUs on a single machine.

   ```bash
   tools/dist_test.sh configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth 4
   ```

### Launching Multiple Tasks on Single Machine

For a workstation equipped with multiple GPUs, the user can launch multiple tasks simultaneously by specifying the GPU IDs. For example, the following command demonstrates how to test DBNet with GPU `[0, 1, 2, 3]` and train CRNN on GPU `[4, 5, 6, 7]`.

```bash
# Specify gpu:0,1,2,3 for testing and assign port number 29500
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_test.sh configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth 4

# Specify gpu:4,5,6,7 for training and assign port number 29501
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/textrecog/crnn/crnn_academic_dataset.py 4
```

```{note}
`dist_train.sh` sets `MASTER_PORT` to `29500` by default. When other processes already occupy this port, the program will get a runtime error `RuntimeError: Address already in use`. In this case, you need to set `MASTER_PORT` to another free port number in the range of `(0~65535)`.
```

### Multi-machine Multi-GPU Training and Testing

You can launch a task on multiple machines connected to the same network. MMOCR relies on `torch.distributed` package for distributed training. Find more information at PyTorch’s [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).

1. **Training**

   The following command demonstrates how to train DBNet on two machines with a total of 4 GPUs.

   ```bash
   # Say that you want to launch the training job on two machines
   # On the first machine:
   NNODES=2 NODE_RANK=0 PORT=29500 MASTER_ADDR=10.140.0.169 tools/dist_train.sh configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py 2
   # On the second machine:
   NNODES=2 NODE_RANK=1 PORT=29501 MASTER_ADDR=10.140.0.169 tools/dist_train.sh configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py 2
   ```

2. **Testing**

   The following command demonstrates how to test DBNet on two machines with a total of 4 GPUs.

   ```bash
   # Say that you want to launch the testing job on two machines
   # On the first machine:
   NNODES=2 NODE_RANK=0 PORT=29500 MASTER_ADDR=10.140.0.169 tools/dist_test.sh configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth 2
   # On the second machine:
   NNODES=2 NODE_RANK=1 PORT=29501 MASTER_ADDR=10.140.0.169 tools/dist_test.sh configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth 2
   ```

   ```{note}
   The speed of the network could be the bottleneck of training.
   ```

## Training and Testing with Slurm Cluster

If you run MMOCR on a cluster managed with [Slurm](https://slurm.schedmd.com/), you can use the script `tools/slurm_train.sh` and `tools/slurm_test.sh`.

```bash
# tools/slurm_train.sh provides scripts for submitting training tasks on clusters managed by the slurm
GPUS=${GPUS} GPUS_PER_NODE=${GPUS_PER_NODE} CPUS_PER_TASK=${CPUS_PER_TASK} SRUN_ARGS=${SRUN_ARGS} ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} [PY_ARGS]

# tools/slurm_test.sh provides scripts for submitting testing tasks on clusters managed by the slurm
GPUS=${GPUS} GPUS_PER_NODE=${GPUS_PER_NODE} CPUS_PER_TASK=${CPUS_PER_TASK} SRUN_ARGS=${SRUN_ARGS} ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${WORK_DIR} [PY_ARGS]
```

| ARGS            | Type | Description                                                                                                 |
| --------------- | ---- | ----------------------------------------------------------------------------------------------------------- |
| GPUS            | int  | The number of GPUs to be used by this task. Defaults to 8.                                                  |
| GPUS_PER_NODE   | int  | The number of GPUs to be allocated per node. Defaults to 8.                                                 |
| CPUS_PER_TASK   | int  | The number of CPUs to be allocated per task. Defaults to 5.                                                 |
| SRUN_ARGS       | str  | Arguments to be parsed by srun. Available options can be found [here](https://slurm.schedmd.com/srun.html). |
| PARTITION       | str  | (required)Specify the partition on cluster.                                                                 |
| JOB_NAME        | str  | (required)Name of the submitted job.                                                                        |
| WORK_DIR        | str  | (required)Specify the working directory for saving the logs and checkpoints.                                |
| CHECKPOINT_FILE | str  | (required，only used in slurm_test.sh)Path to the checkpoint to be tested.                                  |
| PY_ARGS         | str  | Arguments to be parsed by `tools/train.py` and `tools/test.py`.                                             |

These scripts enable training and testing on slurm clusters, see the following examples.

1. Training

   Here is an example of using 1 GPU to train a DBNet model on the `dev` partition.

   ```bash
   # Example: Request 1 GPU resource on dev partition for DBNet training task
   GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=5 tools/slurm_train.sh dev db_r50 configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py work_dir
   ```

2. Testing

   Similarly, the following example requests 1 GPU for testing.

   ```bash
   # Example: Request 1 GPU resource on dev partition for DBNet testing task
   GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=5 tools/slurm_test.sh dev db_r50 configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth work_dir
   ```

## Advanced Tips

### Resume Training from a Checkpoint

`tools/train.py` allows users to resume training from a checkpoint by specifying the `--resume` parameter, where it will automatically resume training from the latest saved checkpoint.

```bash
# Example: Resuming training from the latest checkpoint
python tools/train.py configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py 4 --resume
```

By default, the program will automatically resume training from the last successfully saved checkpoint in the last training session, i.e. `latest.pth`. However,

```python
# Example: Set the path of the checkpoint you want to load in the configuration file
load_from = 'work_dir/dbnet/models/epoch_10000.pth'
```

### Mixed Precision Training

Mixed precision training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. In MMOCR, the users can enable the automatic mixed precision training by simply add `--amp`.

```bash
# Example: Using automatic mixed precision training
python tools/train.py configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py 4 --amp
```

The following table shows the support of each algorithm in MMOCR for automatic mixed precision training.

|               | Whether support AMP |               Description               |
| ------------- | :-----------------: | :-------------------------------------: |
|               |   Text Detection    |                                         |
| DBNet         |          Y          |                                         |
| DBNetpp       |          Y          |                                         |
| DRRG          |          N          | roi_align_rotated does not support fp16 |
| FCENet        |          N          |      BCELoss does not support fp16      |
| Mask R-CNN    |          Y          |                                         |
| PANet         |          Y          |                                         |
| PSENet        |          Y          |                                         |
| TextSnake     |          N          |                                         |
|               |  Text Recognition   |                                         |
| ABINet        |          Y          |                                         |
| CRNN          |          Y          |                                         |
| MASTER        |          Y          |                                         |
| NRTR          |          Y          |                                         |
| RobustScanner |          Y          |                                         |
| SAR           |          Y          |                                         |
| SATRN         |          Y          |                                         |

### Automatic Learning Rate Scaling

MMOCR sets default initial learning rates for each model in the configuration file. However, these initial learning rates may not be applicable when the user uses a different `batch_size` than our preset `base_batch_size`. Therefore, we provide a tool to automatically scale the learning rate, which can be called by adding the `--auto-scale-lr`.

```bash
# Example: Using automatic learning rate scaling
python tools/train.py configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py 4 --auto-scale-lr
```

### Visualize the Predictions

`tools/test.py` provides the visualization interface to facilitate the qualitative analysis of the OCR models.

<div align="center">

![Detection](../../../demo/resources/det_vis.png)

(Green boxes are GTs, while red boxes are predictions)

</div>

<div align="center">

![Recognition](../../../demo/resources/rec_vis.png)

(Green font is the GT, red font is the prediction)

</div>

<div align="center">

![KIE](../../../demo/resources/kie_vis.png)

(From left to right: original image, text detection and recognition result, text classification result, relationship)

</div>

```bash
# Example 1: Show the visualization results per 2 seconds
python tools/test.py configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth --show --wait-time 2

# Example 2: For systems that do not support graphical interfaces (such as computing clusters, etc.), the visualization results can be dumped in the specified path
python tools/test.py configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth --show-dir ./vis_results
```

The visualization-related parameters in `tools/test.py` are described as follows.

| ARGS        | Type  | Description                                   |
| ----------- | ----- | --------------------------------------------- |
| --show      | bool  | Whether to show the visualization results.    |
| --show-dir  | str   | Path to save the visualization results.       |
| --wait-time | float | Interval of visualization (s), defaults to 2. |
