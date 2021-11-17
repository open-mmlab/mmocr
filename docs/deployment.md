# Deployment

We provide deployment tools under `tools/deployment` directory.

## Convert to ONNX (experimental)

We provide a script to convert the model to [ONNX](https://github.com/onnx/onnx) format. The converted model could be visualized by tools like [Netron](https://github.com/lutzroeder/netron). Besides, we also support comparing the output results between PyTorch and ONNX model.

```bash
python tools/deployment/pytorch2onnx.py
    ${MODEL_CONFIG_PATH} \
    ${MODEL_CKPT_PATH} \
    ${MODEL_TYPE} \
    ${IMAGE_PATH} \
    --output-file ${OUTPUT_FILE} \
    --device-id ${DEVICE_ID} \
    --opset-version ${OPSET_VERSION} \
    --verify \
    --verbose \
    --show \
    --dynamic-export
```

Description of arguments:

| ARGS               | Type           | Description                                                                                        |
| ------------------ | -------------- | -------------------------------------------------------------------------------------------------- |
| `model_config`     | str            | The path to a model config file.                                                                   |
| `model_ckpt`       | str            | The path to a model checkpoint file.                                                               |
| `model_type`       | 'recog', 'det' | The model type of the config file.                                                                 |
| `image_path`       | str            | The path to input image file.                                                                      |
| `--output-file`    | str            | The path to output ONNX model. Defaults to `tmp.onnx`.                                             |
| `--device-id`      | int            | Which GPU to use. Defaults to 0.                                                                   |
| `--opset-version`  | int            | ONNX opset version. Defaults to 11.                                                                |
| `--verify`         | bool           | Determines whether to verify the correctness of an exported model. Defaults to `False`.            |
| `--verbose`        | bool           | Determines whether to print the architecture of the exported model. Defaults to `False`.           |
| `--show`           | bool           | Determines whether to visualize outputs of ONNXRuntime and PyTorch. Defaults to `False`.           |
| `--dynamic-export` | bool           | Determines whether to export ONNX model with dynamic input and output shapes. Defaults to `False`. |

:::{note}
This tool is still experimental. For now, some customized operators are not supported, and we only support a subset of detection and recognition algorithms.
:::

### List of supported models exportable to ONNX

The table below lists the models that are guaranteed to be exportable to ONNX and runnable in ONNX Runtime.

| Model  |                                                                      Config                                                                      | Dynamic Shape | Batch Inference |                  Note                  |
| :----: | :----------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :-------------: | :------------------------------------: |
| DBNet  |    [dbnet_r18_fpnc_1200e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py)    |       Y       |        N        |                                        |
| PSENet |     [psenet_r50_fpnf_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py)      |       Y       |        Y        |                                        |
| PSENet |   [psenet_r50_fpnf_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py)    |       Y       |        Y        |                                        |
| PANet  |   [panet_r18_fpem_ffm_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py)   |       Y       |        Y        |                                        |
| PANet  | [panet_r18_fpem_ffm_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |       Y       |        Y        |                                        |
|  CRNN  |            [crnn_academic_dataset.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/crnn_academic_dataset.py)             |       Y       |        Y        | CRNN only accepts input with height 32 |

:::{note}
- *All models above are tested with PyTorch==1.8.1 and onnxruntime-gpu == 1.8.1*
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmocr`.
:::

## Convert ONNX to TensorRT (experimental)

We also provide a script to convert [ONNX](https://github.com/onnx/onnx) model to [TensorRT](https://github.com/NVIDIA/TensorRT) format. Besides, we support comparing the output results between ONNX and TensorRT model.


```bash
python tools/deployment/onnx2tensorrt.py
    ${MODEL_CONFIG_PATH} \
    ${MODEL_TYPE} \
    ${IMAGE_PATH} \
    ${ONNX_FILE} \
    --trt-file ${OUT_TENSORRT} \
    --max-shape INT INT INT INT \
    --min-shape INT INT INT INT \
    --workspace-size INT \
    --fp16 \
    --verify \
    --show \
    --verbose
```

Description of arguments:

| ARGS               | Type           | Description                                                                                         |
| ------------------ | -------------- | --------------------------------------------------------------------------------------------------- |
| `model_config`     | str            | The path to a model config file.                                                                    |
| `model_type`       | 'recog', 'det' | The model type of the config file.                                                                  |
| `image_path`       | str            | The path to input image file.                                                                       |
| `onnx_file`        | str            | The path to input ONNX file.                                                                        |
| `--trt-file`       | str            | The path of output TensorRT model. Defaults to `tmp.trt`.                                           |
| `--max-shape`      | int * 4        | Maximum shape of model input.                                                                       |
| `--min-shape`      | int * 4        | Minimum shape of model input.                                                                       |
| `--workspace-size` | int            | Max workspace size in GiB. Defaults to 1.                                                           |
| `--fp16`           | bool           | Determines whether to export TensorRT with fp16 mode. Defaults to `False`.                          |
| `--verify`         | bool           | Determines whether to verify the correctness of an exported model. Defaults to `False`.             |
| `--show`           | bool           | Determines whether to show the output of ONNX and TensorRT. Defaults to `False`.                    |
| `--verbose`        | bool           | Determines whether to verbose logging messages while creating TensorRT engine. Defaults to `False`. |

:::{note}
This tool is still experimental. For now, some customized operators are not supported, and we only support a subset of detection and recognition algorithms.
:::

### List of supported models exportable to TensorRT

The table below lists the models that are guaranteed to be exportable to TensorRT engine and runnable in TensorRT.

| Model  |                                                                      Config                                                                      | Dynamic Shape | Batch Inference |                  Note                  |
| :----: | :----------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :-------------: | :------------------------------------: |
| DBNet  |    [dbnet_r18_fpnc_1200e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py)    |       Y       |        N        |                                        |
| PSENet |     [psenet_r50_fpnf_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py)      |       Y       |        Y        |                                        |
| PSENet |   [psenet_r50_fpnf_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py)    |       Y       |        Y        |                                        |
| PANet  |   [panet_r18_fpem_ffm_600e_ctw1500.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py)   |       Y       |        Y        |                                        |
| PANet  | [panet_r18_fpem_ffm_600e_icdar2015.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |       Y       |        Y        |                                        |
|  CRNN  |            [crnn_academic_dataset.py](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/crnn_academic_dataset.py)             |       Y       |        Y        | CRNN only accepts input with height 32 |

:::{note}
- *All models above are tested with PyTorch==1.8.1,  onnxruntime-gpu==1.8.1 and tensorrt==7.2.1.6*
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmocr`.
:::


## Evaluate ONNX and TensorRT Models (experimental)

We provide methods to evaluate TensorRT and ONNX models in `tools/deployment/deploy_test.py`.

### Prerequisite
To evaluate ONNX and TensorRT models, ONNX, ONNXRuntime and TensorRT should be installed first. Install `mmcv-full` with ONNXRuntime custom ops and TensorRT plugins follow [ONNXRuntime in mmcv](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) and [TensorRT plugin in mmcv](https://github.com/open-mmlab/mmcv/blob/master/docs/tensorrt_plugin.md).

### Usage

```bash
python tools/deploy_test.py \
    ${CONFIG_FILE} \
    ${MODEL_PATH} \
    ${MODEL_TYPE} \
    ${BACKEND} \
    --eval ${METRICS} \
    --device ${DEVICE}
```

### Description of all arguments

| ARGS           | Type                      | Description                                                                             |
| -------------- | ------------------------- | --------------------------------------------------------------------------------------- |
| `model_config` | str                       | The path to a model config file.                                                        |
| `model_file`   | str                       | The path to a TensorRT or an ONNX model file.                                           |
| `model_type`   | 'recog', 'det'            | Detection or recognition model to deploy.                                               |
| `backend`      | 'TensorRT', 'ONNXRuntime' | The backend for testing.                                                                |
| `--eval`       | 'acc', 'hmean-iou'        | The evaluation metrics. 'acc' for recognition models, 'hmean-iou' for detection models. |
| `--device`     | str                       | Device for evaluation. Defaults to `cuda:0`.                                            |

## Results and Models


<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8">Model</th>
    <th class="tg-9wq8">Config</th>
    <th class="tg-9wq8">Dataset</th>
    <th class="tg-9wq8">Metric</th>
    <th class="tg-9wq8">PyTorch</th>
    <th class="tg-9wq8">ONNX Runtime</th>
    <th class="tg-9wq8">TensorRT FP32</th>
    <th class="tg-9wq8">TensorRT FP16</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="3">DBNet</td>
    <td class="tg-9wq8" rowspan="3">dbnet_r18_fpnc_1200e_icdar2015.py<br></td>
    <td class="tg-9wq8" rowspan="3">icdar2015</td>
    <td class="tg-9wq8"><span style="font-style:normal">Recall</span><br></td>
    <td class="tg-9wq8">0.731</td>
    <td class="tg-9wq8">0.731</td>
    <td class="tg-9wq8">0.678</td>
    <td class="tg-9wq8">0.679</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision</td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.871</span></td>
    <td class="tg-9wq8">0.871</td>
    <td class="tg-9wq8">0.844</td>
    <td class="tg-9wq8">0.842</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><span style="font-style:normal">Hmean</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.795</span></td>
    <td class="tg-9wq8">0.795</td>
    <td class="tg-9wq8">0.752</td>
    <td class="tg-9wq8">0.752</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">DBNet*</td>
    <td class="tg-9wq8" rowspan="3">dbnet_r18_fpnc_1200e_icdar2015.py<br></td>
    <td class="tg-9wq8" rowspan="3">icdar2015</td>
    <td class="tg-9wq8"><span style="font-style:normal">Recall</span><br></td>
    <td class="tg-9wq8">0.720</td>
    <td class="tg-9wq8">0.720</td>
    <td class="tg-9wq8">0.720</td>
    <td class="tg-9wq8">0.718</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision</td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.868</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.868</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.868</span></td>
    <td class="tg-9wq8">0.868</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><span style="font-style:normal">Hmean</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.787</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.787</span></td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">0.787</span></td>
    <td class="tg-9wq8">0.786</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">PSENet</td>
    <td class="tg-9wq8" rowspan="3">psenet_r50_fpnf_600e_icdar2015.py<br></td>
    <td class="tg-9wq8" rowspan="3">icdar2015</td>
    <td class="tg-9wq8"><span style="font-style:normal">Recall</span><br></td>
    <td class="tg-9wq8">0.753</td>
    <td class="tg-9wq8">0.753</td>
    <td class="tg-9wq8">0.753</td>
    <td class="tg-9wq8">0.752</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision</td>
    <td class="tg-9wq8">0.867</td>
    <td class="tg-9wq8">0.867</td>
    <td class="tg-9wq8">0.867</td>
    <td class="tg-9wq8">0.867</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><span style="font-style:normal">Hmean</span></td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.805</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">PANet</td>
    <td class="tg-9wq8" rowspan="3">panet_r18_fpem_ffm_600e_icdar2015.py<br></td>
    <td class="tg-9wq8" rowspan="3">icdar2015</td>
    <td class="tg-9wq8">Recall<br></td>
    <td class="tg-9wq8">0.740</td>
    <td class="tg-9wq8">0.740</td>
    <td class="tg-9wq8">0.687</td>
    <td class="tg-9wq8">N/A</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision</td>
    <td class="tg-9wq8">0.860</td>
    <td class="tg-9wq8">0.860</td>
    <td class="tg-9wq8">0.815</td>
    <td class="tg-9wq8">N/A</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Hmean</td>
    <td class="tg-9wq8">0.796</td>
    <td class="tg-9wq8">0.796</td>
    <td class="tg-9wq8">0.746</td>
    <td class="tg-9wq8">N/A</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="3">PANet*</td>
    <td class="tg-nrix" rowspan="3">panet_r18_fpem_ffm_600e_icdar2015.py<br></td>
    <td class="tg-nrix" rowspan="3">icdar2015</td>
    <td class="tg-nrix">Recall<br></td>
    <td class="tg-nrix">0.736</td>
    <td class="tg-nrix">0.736</td>
    <td class="tg-nrix">0.736</td>
    <td class="tg-nrix">N/A</td>
  </tr>
  <tr>
    <td class="tg-nrix">Precision</td>
    <td class="tg-nrix">0.857</td>
    <td class="tg-nrix">0.857</td>
    <td class="tg-nrix">0.857</td>
    <td class="tg-nrix">N/A</td>
  </tr>
  <tr>
    <td class="tg-nrix">Hmean</td>
    <td class="tg-nrix">0.792</td>
    <td class="tg-nrix">0.792</td>
    <td class="tg-nrix">0.792</td>
    <td class="tg-nrix">N/A</td>
  </tr>
  <tr>
    <td class="tg-9wq8">CRNN</td>
    <td class="tg-9wq8">crnn_academic_dataset.py<br></td>
    <td class="tg-9wq8">IIIT5K</td>
    <td class="tg-9wq8">Acc</td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.806</td>
    <td class="tg-9wq8">0.806</td>
  </tr>
</tbody>
</table>

:::{note}
- TensorRT upsampling operation is a little different from PyTorch. For DBNet and PANet, we suggest replacing upsampling operations with the nearest mode to operations with bilinear mode. [Here](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpem_ffm.py#L33) for PANet, [here](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpn_cat.py#L111) and [here](https://github.com/open-mmlab/mmocr/blob/50a25e718a028c8b9d96f497e241767dbe9617d1/mmocr/models/textdet/necks/fpn_cat.py#L121) for DBNet. As is shown in the above table, networks with tag * mean the upsampling mode is changed.
- Note that changing upsampling mode reduces less performance compared with using the nearest mode. However, the weights of networks are trained through the nearest mode. To pursue the best performance, using bilinear mode for both training and TensorRT deployment is recommended.
- All ONNX and TensorRT models are evaluated with dynamic shapes on the datasets, and images are preprocessed according to the original config file.
- This tool is still experimental, and we only support a subset of detection and recognition algorithms for now.
:::


## C++ Inference example with OpenCV
The example below is tested with Visual Studio 2019 as console application, CPU inference only.

### Prerequisites

1. Project should use OpenCV (tested with version 4.5.4), ONNX Runtime NuGet package (version 1.9.0).
2. Download *DBNet_r18* detector and *SATRN_small* recognizer models from our [Model Zoo](modelzoo.md), and export them with the following python commands (you may change the paths accordingly):

```bash
python3.9 ../mmocr/tools/deployment/pytorch2onnx.py --verify --output-file detector.onnx ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py ./dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth --dynamic-export det ./sample_big_image_eg_1920x1080.png

python3.9 ../mmocr/tools/deployment/pytorch2onnx.py --opset 14 --verify --output-file recognizer.onnx ../mmocr/configs/textrecog/satrn/satrn_small.py ./satrn_small_20211009-2cf13355.pth recog ./sample_small_image_eg_200x50.png
```

:::{note}
- Be aware, while exported `detector.onnx` file is relatively small (about 50 Mb), `recognizer.onnx` is pretty big (more than 600 Mb).
- *DBNet_r18* can use ONNX opset 11, *SATRN_small* can be exported with opset 14.
:::

:::{warning}
Be sure, that verifications of both models are successful - look through the export messages.
:::

### Example
Example usage of exported models with C++ is in the code below (don't forget to change paths to \*.onnx files). It's applicable to these two models only, other models have another preprocessing and postprocessing logics.

```C++
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <onnxruntime_cxx_api.h>
#pragma comment(lib, "onnxruntime.lib")

// DB_r18
class Detector {
public:
	Detector(const std::string& model_path) {
		session = Ort::Session{ env, std::wstring(model_path.begin(), model_path.end()).c_str(), Ort::SessionOptions{nullptr} };
	}

	std::vector<cv::Rect> inference(const cv::Mat& original, float threshold = 0.3f) {

		cv::Size original_size = original.size();

		const char* input_names[] = { "input" };
		const char* output_names[] = { "output" };

		std::array<int64_t, 4> input_shape{ 1, 3, height, width };

		cv::Mat image = cv::Mat::zeros(cv::Size(width, height), original.type());
		cv::resize(original, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);

		image.convertTo(image, CV_32FC3);

		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
		image = (image - cv::Scalar(123.675f, 116.28f, 103.53f)) / cv::Scalar(58.395f, 57.12f, 57.375f);

		cv::Mat blob = cv::dnn::blobFromImage(image);

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, blob.total(), input_shape.data(), input_shape.size());

		std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);

		int sizes[] = { 1, 3, height, width };
		cv::Mat output(4, sizes, CV_32F, output_tensor.front().GetTensorMutableData<float>());

		std::vector<cv::Mat> images;
		cv::dnn::imagesFromBlob(output, images);

		std::vector<cv::Rect> areas = get_detected(images[0], threshold);
		std::vector<cv::Rect> results;

		float x_ratio = original_size.width / (float)width;
		float y_ratio = original_size.height / (float)height;

		for (int index = 0; index < areas.size(); ++index) {
			cv::Rect box = areas[index];

			box.x = int(box.x * x_ratio);
			box.width = int(box.width * x_ratio);
			box.y = int(box.y * y_ratio);
			box.height = int(box.height * y_ratio);

			results.push_back(box);
		}

		return results;
	}

private:
	Ort::Env env;
	Ort::Session session{ nullptr };

	const int width = 1312, height = 736;

	cv::Rect expand_box(const cv::Rect& original, int addition = 5) {
		cv::Rect box(original);
		box.x = std::max(0, box.x - addition);
		box.y = std::max(0, box.y - addition);
		box.width = (box.x + box.width + addition * 2 > width) ? (width - box.x) : (box.width + addition * 2);
		box.height = (box.y + box.height + addition * 2) > height ? (height - box.y) : (box.height + addition * 2);
		return box;
	}

	std::vector<cv::Rect> get_detected(const cv::Mat& output, float threshold) {
		cv::Mat text_mask = cv::Mat::zeros(height, width, CV_32F);
		std::vector<cv::Mat> maps;
		cv::split(output, maps);
		cv::Mat proba_map = maps[0];

		cv::threshold(proba_map, text_mask, threshold, 1.0f, cv::THRESH_BINARY);
		cv::multiply(text_mask, 255, text_mask);
		text_mask.convertTo(text_mask, CV_8U);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(text_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		std::vector<cv::Rect> boxes;

		for (int index = 0; index < contours.size(); ++index) {
			cv::Rect box = expand_box(cv::boundingRect(contours[index]));
			boxes.push_back(box);
		}

		return boxes;
	}
};

// SATRN_small
class Recognizer {
public:
	Recognizer(const std::string& model_path) {
		session = Ort::Session{ env, std::wstring(model_path.begin(), model_path.end()).c_str(), Ort::SessionOptions{nullptr} };
	}

	std::string inference(const cv::Mat& original) {
		const char* input_names[] = { "input" };
		const char* output_names[] = { "output" };

		std::array<int64_t, 4> input_shape{ 1, 3, height, width };

		cv::Mat image;
		cv::resize(original, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
		image.convertTo(image, CV_32FC3);

		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
		image = (image / 255.0f - cv::Scalar(0.485f, 0.456f, 0.406f)) / cv::Scalar(0.229f, 0.224f, 0.225f);

		cv::Mat blob = cv::dnn::blobFromImage(image);

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, blob.total(), input_shape.data(), input_shape.size());

		std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);

		int sequence_length = 25;
		std::string dictionary = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]_`~";
		int characters = dictionary.length() + 2; // EOS + UNK

		std::vector<int> max_indices;
		for (int outer = 0; outer < sequence_length; ++outer) {
			int character_index = -1;
			float character_value = 0;
			for (int inner = 0; inner < characters; ++inner) {
				int counter = outer * characters + inner;
				float value = output_tensor[0].GetTensorMutableData<float>()[counter];
				if (value > character_value) {
					character_value = value;
					character_index = inner;
				}
			}
			max_indices.push_back(character_index);
		}

		std::string recognized;

		for (int index = 0; index < max_indices.size(); ++index) {
			if (max_indices[index] == dictionary.length()) {
				continue; // unk
			}
			if (max_indices[index] == dictionary.length() + 1) {
				break; // eos
			}
			recognized += dictionary[max_indices[index]];
		}

		return recognized;
	}

private:
	Ort::Env env;
	Ort::Session session{ nullptr };

	const int height = 32;
	const int width = 100;
};

int main(int argc, const char* argv[]) {
	if (argc < 2) {
		std::cout << "Usage: this_executable.exe c:/path/to/image.png" << std::endl;
		return 0;
	}

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	std::cout << "Loading models..." << std::endl;

	Detector detector("d:/path/to/detector.onnx");
	Recognizer recognizer("d:/path/to/recognizer.onnx");

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Loading models done in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

	cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

	begin = std::chrono::steady_clock::now();
	std::vector<cv::Rect> detections = detector.inference(image);
	for (int index = 0; index < detections.size(); ++index) {
		cv::Mat roi = image(detections[index]);
		std::string text = recognizer.inference(roi);
		cv::rectangle(image, detections[index], cv::Scalar(255, 255, 255), 2);
		cv::putText(image, text, cv::Point(detections[index].x, detections[index].y - 10), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 255));
	}

	end = std::chrono::steady_clock::now();
	std::cout << "Inference time (with drawing): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

	cv::imshow("Results", image);
	cv::waitKey(0);

	return 0;
}
```

The output should look something like this.
```
Loading models...
Loading models done in 5715 ms
Inference time (with drawing): 3349 ms
```

And the sample result should look something like this.
![resultspng](https://user-images.githubusercontent.com/93123994/142095495-40400ec9-875e-403d-98fa-0a52da385269.png)
