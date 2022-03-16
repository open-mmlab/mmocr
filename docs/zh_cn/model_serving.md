# 模型服务

`MMOCR` 提供了一些便利可以促进模型服务处理。下面是一个快速的演练，通过 API 提供模型服务的一些必要步骤。

## 安装 TorchServe

你可以根据[官网](https://github.com/pytorch/serve#install-torchserve-and-torch-model-archiver)步骤来安装 `TorchServe` 和
`torch-model-archiver` 两个模块。

##  将 MMOCR 模型转换为 TorchServe

我们提供了一个便捷的工具可以将任何以 `.pth` 为后缀的模型转换为以 `.mar` 结尾的模型来满足 TorchServe 使用要求。

```shell
python tools/deployment/mmocr2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output-folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

:::{note}
${MODEL_STORE} 必须是文件夹的绝对路径。
:::

例如：

```shell
python tools/deployment/mmocr2torchserve.py \
  configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py \
  checkpoints/dbnet_r18_fpnc_1200e_icdar2015.pth \
  --output-folder ./checkpoints \
  --model-name dbnet
```

## 启动服务

### 本地启动

在准备好模型后，下一步是使用命令行启动服务：

```bash
# 加载所有的模型到 ./checkpoints
torchserve --start --model-store ./checkpoints --models all
# 或者你仅仅使用一个模型服务，比如 dbnet
torchserve --start --model-store ./checkpoints --models dbnet=dbnet.mar
```

然后，你可以通过 TorchServe 的 REST API 访问推理、管理和指标等服务。你可以在[TorchServe REST API](https://github.com/pytorch/serve/blob/master/docs/rest_api.md) 中找到它们的用法。


| 服务           |  地址                                                            |
| ------------------- | ----------------------- |
| 推理 | `http://127.0.0.1:8080` |
| 管理 | `http://127.0.0.1:8081` |
| 指标 | `http://127.0.0.1:8082` |

:::{note}
TorchServe 默认会将服务绑定到端口 `8080`、 `8081` 、 `8082` 上。你可以通过修改 `config.properties` 来更改端口及存储位置等内容，并通过可选项 `--ts-config config.preperties` 来运行 TorchServe 服务。

```bash
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=32
job_queue_size=1000
model_store=/home/model-server/model-store
```

:::


### 通过 Docker 启动

通过 Docker 提供模型服务不失为一种更好的方法。我们提供了一个 Dockerfile，可以让你摆脱那些繁琐且容易出错的环境设置步骤。

#### 构建 `mmocr-serve` Docker 镜像

```shell
docker build -t mmocr-serve:latest docker/serve/
```

#### 通过 Docker 运行 `mmocr-serve`

为了在 GPU 环境下运行 Docker， 首先需要安装 [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)；或者你也可以只使用 CPU 环境而不必加  `--gpus` 参数。

下面的命令将使用 gpu 运行，将推断、管理、指标的端口绑定到8080、8081、8082上，将容器的IP绑定到127.0.0.1上，并将检查点文件夹 `./checkpoints` 从主机挂载到容器的 `/home/model-server/model-store` 文件夹下。更多相关信息，请查看官方文档中 [docker中运行 TorchServe 服务](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment)。

```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=`realpath ./checkpoints`,target=/home/model-server/model-store \
mmocr-serve:latest
```

:::{note}
`realpath ./checkpoints` 指向的是 "./checkpoints" 的绝对路径，你也可以将其替换为你的 torchserve 模型所在的绝对路径。
:::

运行docker后，你可以通过 TorchServe 的 REST API 访问推理、管理和指标服务。具体你可以在[TorchServe REST API](https://github.com/pytorch/serve/blob/master/docs/rest_api.md) 中找到它们的用法。

| 服务           |  地址                                                            |
| ------------------- | ----------------------- |
| 推理 | `http://127.0.0.1:8080` |
| 管理 | `http://127.0.0.1:8081` |
| 指标 | `http://127.0.0.1:8082` |



## 4. 测试部署

推理 API 允许用户发送一张图到模型服务中，并返回相应的预测结果。

```shell
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T demo/demo_text_det.jpg
```

例如，

```shell
curl http://127.0.0.1:8080/predictions/dbnet -T demo/demo_text_det.jpg
```

对于检测模型，你会获取到名为 `boundary_result` 的 json 对象。内部的每个数组都有浮点数，边界顶点的 x、y 坐标以顺时针顺序表示，最后一个浮点数作为置信度分数。

```json
{
  "boundary_result": [
    [
      221.18990004062653,
      226.875,
      221.18990004062653,
      212.625,
      244.05868631601334,
      212.625,
      244.05868631601334,
      226.875,
      0.80883354575186
    ]
  ]
}
```

对于识别模型，返回的结果如下：

```json
{
  "text": "sier",
  "score": 0.5247521847486496
}
```

同时可以使用 `test_torchserve.py` 来可视化对比 TorchServe 和 PyTorch 结果。

```shell
python tools/deployment/test_torchserve.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MODEL_NAME}
[--inference-addr ${INFERENCE_ADDR}] [--device ${DEVICE}]
```

例如：

```shell
python tools/deployment/test_torchserve.py \
  demo/demo_text_det.jpg \
  configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py \
  checkpoints/dbnet_r18_fpnc_1200e_icdar2015.pth \
  dbnet
```
