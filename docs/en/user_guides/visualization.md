# Visualization

Before reading this tutorial, it is recommended to read MMEngine's {external+mmengine:doc}`MMEngine: Visualization <advanced_tutorials/visualization>` documentation to get a first glimpse of the `Visualizer` definition and usage.

In brief, the [`Visualizer`](mmengine.visualization.Visualizer) is implemented in MMEngine to meet the daily visualization needs, and contains three main functions:

- Implement common drawing APIs, such as [`draw_bboxes`](mmengine.visualization.Visualizer.draw_bboxes) which implements bounding box drawing functions, [`draw_lines`](mmengine.visualization.Visualizer.draw_lines) implements the line drawing function.
- Support writing visualization results, learning rate curves, loss function curves, and verification accuracy curves to various backends, including local disks and common deep learning training logging tools such as [TensorBoard](https://www.tensorflow.org/tensorboard) and [Wandb](https://wandb.ai/site).
- Support calling anywhere in the code to visualize or record intermediate states of the model during training or testing, such as feature maps and validation results.

Based on MMEngine's Visualizer, MMOCR comes with a variety of pre-built visualization tools that can be used by the user by simply modifying the following configuration files.

- The `tools/analysis_tools/browse_dataset.py` script provides a dataset visualization function that draws images and corresponding annotations after Data Transforms, as described in [`browse_dataset.py`](useful_tools.md).
- MMEngine implements `LoggerHook`, which uses `Visualizer` to write the learning rate, loss and evaluation results to the backend set by `Visualizer`. Therefore, by modifying the `Visualizer` backend in the configuration file, for example to ` TensorBoardVISBackend` or `WandbVISBackend`, you can implement logging to common training logging tools such as `TensorBoard` or `WandB`, thus making it easy for users to use these visualization tools to analyze and monitor the training process.
- The `VisualizerHook` is implemented in MMOCR, which uses the `Visualizer` to visualize or store the prediction results of the validation or prediction phase into the backend set by the `Visualizer`, so by modifying the `Visualizer` backend in the configuration file, for example, to ` TensorBoardVISBackend` or `WandbVISBackend`, you can implement storing the predicted images to `TensorBoard` or `Wandb`.

## Configuration

Thanks to the use of the registration mechanism, in MMOCR we can set the behavior of the `Visualizer` by modifying the configuration file. Usually, we define the default configuration for the visualizer in `task/_base_/default_runtime.py`, see [configuration tutorial](config.md) for details.

```Python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextxxxLocalVisualizer', # use different visualizers for different tasks
    vis_backends=vis_backends,
    name='visualizer')
```

Based on the above example, we can see that the configuration of `Visualizer` consists of two main parts, namely, the type of `Visualizer` and the visualization backend `vis_backends` it uses.

- For different OCR tasks, various visualizers are pre-configured in MMOCR, including [`TextDetLocalVisualizer`](mmocr.visualization.TextDetLocalVisualizer), [`TextRecogLocalVisualizer`](mmocr.visualization.TextRecogLocalVisualizer), [`TextSpottingLocalVisualizer`](mmocr.visualization.TextSpottingLocalVisualizer) and [`KIELocalVisualizer`](mmocr.visualization.KIELocalVisualizer). These visualizers extend the basic Visulizer API according to the characteristics of their tasks and implement the corresponding tag information interface `add_datasamples`. For example, users can directly use `TextDetLocalVisualizer` to visualize labels or predictions for text detection tasks.
- MMOCR sets the visualization backend `vis_backend` to the local visualization backend `LocalVisBackend` by default, saving all visualization results and other training information in a local folder.

## Storage

MMOCR uses the local visualization backend [`LocalVisBackend`](mmengine.visualization.LocalVisBackend) by default, and the model loss, learning rate, model evaluation accuracy and visualization The information stored in `VisualizerHook` and `LoggerHook`, including loss, learning rate, evaluation accuracy will be saved to the `{work_dir}/{config_name}/{time}/{vis_data}` folder by default. In addition, MMOCR also supports other common visualization backends, such as `TensorboardVisBackend` and `WandbVisBackend`, and you only need to change the `vis_backends` type in the configuration file to the corresponding visualization backend. For example, you can store data to `TensorBoard` and `Wandb` by simply inserting the following code block into the configuration file.

```Python
_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend'),]
```

## Plot

### Plot the prediction results

MMOCR mainly uses [`VisualizationHook`](mmocr.engine.hooks.VisualizationHook) to plot the prediction results of validation and test, by default `VisualizationHook` is off, and the default configuration is as follows.

```Python
visualization=dict( # user visualization of validation and test results
    type='VisualizationHook',
    enable=False,
    interval=1,
    show=False,
    draw_gt=False,
    draw_pred=False)
```

The following table shows the parameters supported by `VisualizationHook`.

| Parameters |                                                  Description                                                  |
| :--------: | :-----------------------------------------------------------------------------------------------------------: |
|   enable   |        The VisualizationHook is turned on and off by the enable parameter, which is the default state.        |
|  interval  | Controls how much iteration to store or display the results of a val or test if VisualizationHook is enabled. |
|    show    |                           Controls whether to visualize the results of val or test.                           |
|  draw_gt   |               Whether the results of val or test are drawn with or without labeling information               |
| draw_pred  |                              whether to draw predictions for val or test results                              |

If you want to enable `VisualizationHook` related functions and configurations during training or testing, you only need to modify the configuration, take `dbnet_resnet18_fpnc_1200e_icdar2015.py` as an example, draw annotations and predictions at the same time, and display the images, the configuration can be modified as follows

```Python
visualization = _base_.default_hooks.visualization
visualization.update(
    dict(enable=True, show=True, draw_gt=True, draw_pred=True))
```

<div align=center>
<img src="https://user-images.githubusercontent.com/24622904/187426573-8448c827-1336-4416-aebc-e7fccce362cd.png" height="200"/>
</div>

If you only want to see the predicted result information you can just let `draw_pred=True`

```Python
visualization = _base_.default_hooks.visualization
visualization.update(
    dict(enable=True, show=True, draw_gt=False, draw_pred=True))
```

<div align=center>
<img src="https://user-images.githubusercontent.com/24622904/187428385-e6a23120-6445-4c55-a265-c550da692087.png" height="300"/>
</div>

The `test.py` procedure is further simplified by providing the  `--show` and `--show-dir` parameters to visualize the annotation and prediction results during the test without modifying the configuration.

```Shell
# Show test results
python tools/test.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py dbnet_r18_fpnc_1200e_icdar2015/epoch_400.pth --show

# Specify where to store the prediction results
python tools/test.py configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py dbnet_r18_fpnc_1200e_icdar2015/epoch_400.pth --show-dir imgs/
```

<div align=center>
<img src="https://user-images.githubusercontent.com/24622904/187426573-8448c827-1336-4416-aebc-e7fccce362cd.png" height="200"/>
</div>
