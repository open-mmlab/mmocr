# Inferencers

In OpenMMLab 2.0, all the inference operations are unified into a new inference - `Inferencer`. `Inferencer` is designed to expose a neat and simple API to users, and shares very similar interface across different OpenMMLab libraries.

In MMOCR, Inferencers are constructed in different levels of task abstraction.

- Task-specific Inferencer: Following OpenMMLab 2.0's convention, each fundamental task in MMOCR has its Inferencer, namely `TextDetInferencer`, `TextRecInferencer`, `TextSpottingInferencer`, and `KIEInferencer`. They are designed to perform inference on a single task, and can be chained together to perform inference on a series of tasks. They also share very similar interface, have standard input/output protocol, and overall follow the OpenMMLab 2.0 design.
- [`MMOCRInferencer`](../user_guides/inference.md): We also provide `MMOCRInferencer`, a convenient inference interface only designed for MMOCR. It encapsulates and chains all the Inferencers in MMOCR, so users can use this Inferencer to perform a series of tasks on an image and directly get the final result in an end-to-end manner. *However, it has a relatively different interface from other task-specific Inferencers, and some of standard Inferencer functionalities might be sacrificed for the sake of simplicity.*

For new users, we recommend using [`MMOCRInferencer`](../user_guides/inference.md) to test out different combinations of models.

If you are a developer and wish to integrate the models into your own project, we recommend using task-specific Inferencers, as they are more flexible and standardized, equipped with full functionalities.

This page will introduce the usage of task-specific Inferencers.

## General Usage

Init

### Model Initialization

Every `Inferencer` reserves two arguments, `model` and `weights`, for initialization. and there are many ways to initialize a model for inference.

- `model` takes either the name of a model, or the path to a config file as input. The name of a model is obtained from the model's metafile ([Example](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet/metafile.yml)) indexed from [model-index.yml](https://github.com/open-mmlab/mmocr/blob/1.x/model-index.yml). You can find the list of available name choices [here](../modelzoo.md#weights).

  ```{note}
  For convenience, we abbreviate the names of some commonly-used models in the "Alias" field of its metafile, which Inferencer can use to index a model as well.
  ```

- `weights` accepts the path to a weight file.

There are many ways to initialize a model.

- To infer with MMOCR's pre-trained model,  you can pass its name to `model`. The weights will be automatically downloaded and loaded from OpenMMLab's model zoo. Check [Weights](../modelzoo.md#weights) for available model names.

  ```python
  >>> from mmocr.apis import TextDetInferencer
  >>> inferencer = TextDetInferencer(model='DBNet')
  ```

  ```{note}
  The model type must match the Inferencer type.
  ```

  To load the custom weight, you can also pass its path/url to `weights`.

  ```python
  >>> inferencer = TextDetInferencer(model='DBNet', weights='path/to/dbnet.pth')
  ```

- To load custom config and weight, you can pass the path to the config file to `model` and the path to the weight to `weights`.

  ```python
  >>> inferencer = TextDetInferencer(model='path/to/dbnet_config.py', weights='path/to/dbnet.pth')
  ```

- By default, [MMEngine](https://github.com/open-mmlab/mmengine/) dumps config to the weight. If you have a weight trained on MMEngine, you can also pass the path to the weight file to `weights` without specifying `model`:

  ```python
  >>> # It will raise an error if the config file cannot be found in the weight
  >>> inferencer = TextDetInferencer(weights='path/to/dbnet.pth')
  ```

- Passing config file to `xxx` without specifying the weight path `xxx_weights` will randomly initialize a model.
