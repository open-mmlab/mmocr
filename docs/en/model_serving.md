# Model Serving

`MMOCR` provides some utilities that facilitate the model serving process.
Here is a quick walkthrough of necessary steps that let the models to serve through an API.

## Install TorchServe

You can follow the steps on the [official website](https://github.com/pytorch/serve#install-torchserve-and-torch-model-archiver) to install `TorchServe` and
`torch-model-archiver`.

## Convert model from MMOCR to TorchServe

We provide a handy tool to convert any `.pth` model into `.mar` model
for TorchServe.

```shell
python tools/deployment/mmocr2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output-folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

```{note}
${MODEL_STORE} needs to be an absolute path to a folder.
```

For example:

```shell
python tools/deployment/mmocr2torchserve.py \
  configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py \
  checkpoints/dbnet_r18_fpnc_1200e_icdar2015.pth \
  --output-folder ./checkpoints \
  --model-name dbnet
```

## Start Serving

### From your Local Machine

Getting your models prepared, the next step is to start the service with a one-line command:

```bash
# To load all the models in ./checkpoints
torchserve --start --model-store ./checkpoints --models all
# Or, if you only want one model to serve, say dbnet
torchserve --start --model-store ./checkpoints --models dbnet=dbnet.mar
```

Then you can access inference, management and metrics services
through TorchServe's REST API.
You can find their usages in [TorchServe REST API](https://github.com/pytorch/serve/blob/master/docs/rest_api.md).

| Service           |  Address                                                            |
| ------------------- | ----------------------- |
| Inference | `http://127.0.0.1:8080` |
| Management | `http://127.0.0.1:8081` |
| Metrics | `http://127.0.0.1:8082` |

````{note}
By default, TorchServe binds port number `8080`, `8081` and `8082` to its services.
You can change such behavior by modifying and saving the contents below to `config.properties`, and running TorchServe with option `--ts-config config.preperties`.

```bash
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=32
job_queue_size=1000
model_store=/home/model-server/model-store
```

````


### From Docker

A better alternative to serve your models is through Docker. We provide a Dockerfile
that frees you from those tedious and error-prone environmental setup steps.

#### Build `mmocr-serve` Docker image

```shell
docker build -t mmocr-serve:latest docker/serve/
```

#### Run `mmocr-serve` with Docker

In order to run Docker in GPU, you need to install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html); or you can omit the `--gpus` argument for a CPU-only session.

The command below will run `mmocr-serve` with a gpu, bind the ports of `8080` (inference),
`8081` (management) and `8082` (metrics) from container to `127.0.0.1`, and mount
the checkpoint folder `./checkpoints` from the host machine to `/home/model-server/model-store`
of the container. For more information, please check the official docs for [running TorchServe with docker](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment).

```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=`realpath ./checkpoints`,target=/home/model-server/model-store \
mmocr-serve:latest
```

```{note}
`realpath ./checkpoints` points to the absolute path of "./checkpoints", and you can replace it with the absolute path where you store torchserve models.
```

Upon running the docker, you can access inference, management and metrics services
through TorchServe's REST API.
You can find their usages in [TorchServe REST API](https://github.com/pytorch/serve/blob/master/docs/rest_api.md).

| Service           |  Address                                                            |
| ------------------- | ----------------------- |
| Inference | `http://127.0.0.1:8080` |
| Management | `http://127.0.0.1:8081` |
| Metrics | `http://127.0.0.1:8082` |



## 4. Test deployment

Inference API allows user to post an image to a model and returns the prediction result.

```shell
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T demo/demo_text_det.jpg
```

For example,

```shell
curl http://127.0.0.1:8080/predictions/dbnet -T demo/demo_text_det.jpg
```

For detection models, you should obtain a json with an object named `boundary_result`. Each array inside has float numbers representing x, y
coordinates of boundary vertices in clockwise order, and the last float number as the
confidence score.

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

For recognition models, the response should look like:

```json
{
  "text": "sier",
  "score": 0.5247521847486496
}
```

And you can use `test_torchserve.py` to compare result of TorchServe and PyTorch by visualizing them.

```shell
python tools/deployment/test_torchserve.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MODEL_NAME}
[--inference-addr ${INFERENCE_ADDR}] [--device ${DEVICE}]
```

Example:

```shell
python tools/deployment/test_torchserve.py \
  demo/demo_text_det.jpg \
  configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py \
  checkpoints/dbnet_r18_fpnc_1200e_icdar2015.pth \
  dbnet
```
