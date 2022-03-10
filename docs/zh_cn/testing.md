# 测试

此文档介绍在数据集上测试预训练模型的方法。

## 使用单GPU进行测试

您可以使用 `tools/test.py` 执行单 CPU/GPU 推理。例如，要在 IC15 上评估 DBNet: ( 可以从[Model Zoo]( modelzoo.md )下载预训练模型 )：

```shell
./tools/dist_test.sh configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth --eval hmean-iou
```

下面是脚本的完整用法:

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

:::{注}
默认情况下，MMOCR 更喜欢 GPU 而非 CPU。如果您想在 CPU 上测试模型，请清空 `CUDA_VISIBLE_DEVICES` 或者将其设置为 -1 以使程序对 GPU(s) 不可见。需要注意的是，运行 CPU 测试需要 **MMCV >= 1.4.4**。

```bash
CUDA_VISIBLE_DEVICES= python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

:::



| 参数               | 类型                              | 描述                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------ | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--out`            | str                               | 以pickle格式输出结果文件。                                                                                                                                                                                                                                                                                                                                                   |
| `--fuse-conv-bn`   | bool                              | 所选det模型的自定义配置的路径。                                                                                                                                                                                                                                                                                                                                  |
| `--format-only`    | bool                              | 格式化输出结果文件而不执行评估。 当您想将结果格式化为特定格式并将它们提交到测试服务器时，它很有用。                                                                                                                                                                                                                     |
| `--gpu-id`         | int                               | 要使用的 GPU ID。仅适用于非分布式训练。                                                                                                                                                                                                                                                                                                                           |
| `--eval`           | 'hmean-ic13', 'hmean-iou', 'acc'  | 不同的任务使用不同的评估指标。对于文本检测任务，指标是 'hmean-ic13' 或者 'hmean-iou'。对于文本识别任务，指标是 'acc'。                                                                                                                                                                                                 |
| `--show`           | bool                              | 是否显示结果。                                                                                                                                                                                                                                                                                                                                                               |
| `--show-dir`       | str                               | 将用于保存输出图像的目录。                                                                                                                                                                                                                                                                                                                                      |
| `--show-score-thr` | float                             | 分数阈值 (默认值: 0.3)。                                                                                                                                                                                                                                                                                                                                                        |
| `--gpu-collect`    | bool                              | 是否使用 gpu 收集结果。                                                                                                                                                                                                                                                                                                                                                 |
| `--tmpdir`         | str                               | 用于从多个 workers 收集结果的 tmp 目录，在未指定 gpu-collect 时可用。                                                                                                                                                                                                                                                                  |
| `--cfg-options`    | str                               | 覆盖使用的配置中的一些设置， xxx=yyy 格式的键值对将被合并到配置文件中。如果要覆盖的值是一个列表，它应当是 key ="[a,b]" 或者 key=a,b 的形式。该参数还允许嵌套列表/元组值，例如 key="[(a,b),(c,d)]"。请注意，引号是必需的，并且不允许使用空格。 |
| `--eval-options`   | str                               | 用于评估的自定义选项， xxx=yyy 格式的键值对将是 dataset.evaluate() 函数的 kwargs。                                                                                                                                                                                                                                                                 |
| `--launcher`       | 'none', 'pytorch', 'slurm', 'mpi' | 工作启动器的选项。                                                                                                                                                                                                                                                                                                                                                             |

## 使用多 GPU 进行测试

MMOCR 使用 `MMDistributedDataParallel` 实现 **分布式**测试。

您可以使用以下命令测试具有多个 GPU 的数据集。


```shell
[PORT={PORT}] ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

| 参数 | 类型 | 描述                                                                      |
| --------- | ---- | -------------------------------------------------------------------------------- |
| `PORT`    | int  | 等级为 0 的机器将使用的主端口。默认为 29500。 |
| `PY_ARGS` | str  | 由 `tools/test.py` 解析的参数。                                       |

例如，

```shell
./tools/dist_test.sh configs/example_config.py work_dirs/example_exp/example_model_20200202.pth 1 --eval hmean-iou
```
