# Migration of Data Transforms

## Introduction

In MMOCR version 0.x, we implemented a series of **Data Transform** methods in `mmocr/datasets/pipelines/xxx_transforms.py`. However, these modules are scattered all over the place and lack a standardized design. Therefore, we refactored all the data transform modules in MMOCR version 1.x. According to the task type, they are now defined in `ocr_transforms.py`, `textdet_transforms.py`, and `recog_transforms.py`, respectively, under `mmocr/datasets/transforms`.

Since some of the modules were renamed, merged or separated during the refactoring process, the new interface and default parameters may be inconsistent with the old version. Therefore, this migration guide will introduce how to configure the new data transforms to achieve the identical behavior as the old version.

## Configuration Migration Guide

### Data Formatting Related Data Transforms

1. `Collect` + `CustomFormatBundle` -> `PackTextDetInputs/PackTextRecogInputs`

`PackxxxInputs` implements both `Collect` and `CustomFormatBundle` functions, and no longer has `key` parameters, the generation of training targets is moved to be done in `loss` modules.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
    type='CustomFormatBundle',
    keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
    meta_keys=['img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction'],
    visualize=dict(flag=False, boundary_key='gt_shrink')),
dict(
    type='Collect',
    keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
```

</td><td>

```python
dict(
  type='PackTextDetInputs',
  meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction'))
```

</td></tr>
</thead>
</table>

### Data Augmentation Related Data Transforms

1. `ResizeOCR` -> `Resize`, `RescaleToHeight`, `PadToWidth`

   The original `ResizeOCR` is now split into three data augmentation modules.

   When `keep_aspect_ratio=False`, it is equivalent to `Resize` in version 1.x. Its configuration can be modified as follows.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='ResizeOCR',
  height=32,
  min_width=100,
  max_width=100,
  keep_aspect_ratio=False)
```

</td><td>

```python
dict(
  type='Resize',
  scale=(100, 32),
  keep_ratio=False)
```

</td></tr>
</thead>
</table>

When `keep_aspect_ratio=True` and `max_width=None`. The image will be rescaled to a fixed size alongside the height while keeping the aspect ratio the same as the origin.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='ResizeOCR',
  height=32,
  min_width=32,
  max_width=None,
  width_downsample_ratio = 1.0 / 16
  keep_aspect_ratio=True)
```

</td><td>

```python
dict(
  type='RescaleToHeight',
  height=32,
  min_width=32,
  max_width=None,
  width_divisor=16),
```

</td></tr>
</thead>
</table>

When `keep_aspect_ratio=True` and `max_width` is a fixed value. The image will be rescaled to a fixed size alongside the height while keeping the aspect ratio the same as the origin. Then, the width will be padded or cropped to `max_width`. That is to say, the shape of the output image is always `(height, max_width)`.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='ResizeOCR',
  height=32,
  min_width=32,
  max_width=100,
  width_downsample_ratio = 1.0 / 16,
  keep_aspect_ratio=True)
```

</td><td>

```python
dict(
  type='RescaleToHeight',
  height=32,
  min_width=32,
  max_width=100,
  width_divisor=16),
dict(
  type='PadToWidth',
  width=100)
```

</td></tr>
</thead>
</table>

2. `RandomRotateTextDet` &  `RandomRotatePolyInstances` -> `RandomRotate`

   We implemented all random rotation-related data augmentation in `RandomRotate` in version 1.x. Its default behavior is identical to the `RandomRotateTextDet` in version 0.x.

```{note}
  The default value of 'max_angle' might be different from the old version, so the users are suggested to manually set the number.
```

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(type='RandomRotateTextDet')
```

</td><td>

```python
dict(type='RandomRotate', max_angle=10)
```

</td></tr>
</thead>
</table>

For `RandomRotatePolyInstances`，it is supposed to set `use_canvas=True`。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='RandomRotatePolyInstances',
  rotate_ratio=0.5, # Specify the execution probability
  max_angle=60,
  pad_with_fixed_color=False)
```

</td><td>

```python
# Wrap the data transforms with RandomApply and specify the execution probability
dict(
  type='RandomApply',
  transforms=[
    dict(type='RandomRotate',
    max_angle=60,
    pad_with_fixed_color=False,
    use_canvas=True)],
  prob=0.5) # Specify the execution probability
```

</td></tr>
</thead>
</table>

```{note}
In version 0.x, some data augmentation methods specified execution probability by defining an internal variable 'xxx_ratio', such as 'rotate_ratio', 'crop_ratio', etc. In version 1.x, these parameters have been removed. Now we can use 'RandomApply' to wrap different data transforms and specify their execution probabilities.
```

3. `RandomCropFlip` -> `TextDetRandomCropFlip`

   Currently, only the method name has been changed, and other parameters remain the same.

4. `RandomCropPolyInstances` -> `RandomCrop`

   In MMOCR version 1.x, `crop_ratio` and `instance_key` are removed. The `gt_polygons` is now used as the target for cropping.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='RandomCropPolyInstances',
  instance_key='gt_masks',
  crop_ratio=0.8, # Specify the execution probability
  min_side_ratio=0.3)
```

</td><td>

```python
# Wrap the data transforms with RandomApply and specify the execution probability
dict(
  type='RandomApply',
  transforms=[dict(type='RandomCrop', min_side_ratio=0.3)],
  prob=0.8) # Specify the execution probability
```

</td></tr>
</thead>
</table>

5. `RandomCropInstances` -> `TextDetRandomCrop`

   In MMOCR version 1.x, `crop_ratio` and `instance_key` are removed. The `gt_polygons` is now used as the target for cropping.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='RandomCropInstances',
  target_size=（800，800）,
  instance_key='gt_kernels')
```

</td><td>

```python
dict(
  type='TextDetRandomCrop',
  target_size=（800，800))
```

</td></tr>
</thead>
</table>

6. `EastRandomCrop` -> `RandomCrop` + `Resize` + `Pad`

   The `EastRandomCrop` was implemented by applying cropping, scaling and padding to the input image. Now, the same effect can be achieved by combining three data transforms.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='EastRandomCrop',
  max_tries=10,
  min_crop_side_ratio=0.1,
  target_size=(640, 640))
```

</td><td>

```python
dict(type='RandomCrop', min_side_ratio=0.1),
dict(type='Resize', scale=(640,640), keep_ratio=True),
dict(type='Pad', size=(640,640))
```

</td></tr>
</thead>
</table>

7. `RandomScaling` -> `mmcv.RandomResize`

   The `RandomScaling` is now replaced with `mmcv.RandomResize`.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
 dict(
  type='RandomScaling',
  size=800,
  scale=(0.75, 2.5))
```

</td><td>

```python
dict(
  type='RandomResize',
  scale=(800, 800),
  ratio_range=(0.75, 2.5),
  keep_ratio=True)
```

</td></tr>
</thead>
</table>

```{note}
By default, the data pipeline will search for the corresponding data transforms from the register of the current 'scope', and if that data transform does not exist, it will continue to search in the upstream library, such as MMCV. For example, the 'RandomResize' function is not implemented in mmocr, but it can be directly called in the configuration, as the program will automatically search for it from MMCV. In addition, the user can also specify 'scope' by adding a prefix. For example, 'mmcv.RandomResize' will force the it to use 'RandomResize' implemented in MMCV, which is useful when a method of the same name exists in both upstream and downstream libraries.
```

8. `SquareResizePad` -> `Resize` + `SourceImagePad`

   The `SquareResizePad` implements two branches and uses one of them randomly based on the `pad_ratio`. Specifically, one branch firstly resize the image and then padding it to a certain size; the other branch only resize the image. To enhance the reusability of the different modules, we split this data transform into a combination of `Resize` + `SourceImagePad` in version 1.x, and control the branches via `RandomChoice`.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='SquareResizePad',
  target_size=800,
  pad_ratio=0.6)
```

</td><td>

```python
dict(
  type='RandomChoice',
  transforms=[
    [
      dict(
        type='Resize',
        scale=800,
        keep_ratio=True),
      dict(
        type='SourceImagePad',
        target_scale=800)
    ],
    [
      dict(
        type='Resize',
        scale=800,
        keep_ratio=False)
    ]
  ],
  prob=[0.4, 0.6]), # Probability of selection of two combinations
```

</td></tr>
</thead>
</table>

```{note}
In version 1.x, the random choice wrapper 'RandomChoice' replaces 'OneOfWrapper', allowing random selection of data transform combinations.
```

9. `RandomWrapper` -> `mmcv.RandomApply`

   In version 1.x, the `RandomWrapper` wrapper has been renamed to `RandomApply`, which is used to specify the probability of performing a data transform. And the probability `p` is now named `prob`.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
 dict(
  type='RandomWrapper',
  p=0.25,
  transforms=[
      dict(type='PyramidRescale'),
  ])
```

</td><td>

```python
dict(
  type='RandomApply',
  prob=0.25,
  transforms=[
    dict(type='PyramidRescale'),
  ])
```

</td></tr>
</thead>
</table>

10. `OneOfWrapper` -> `mmcv.RandomChoice`

    The random choice wrapper is now renamed to `RandomChoice` and is used in exactly the same way as before.

11. `ScaleAspectJitter` -> `ShortScaleAspectJitter`, `BoundedScaleAspectJitter`

    The `ScaleAspectJitter` implemented several different image size jittering strategies, which has now been split into several independent data transforms.

    When `resize_type='indep_sample_in_range'`, it is equivalent to `RandomResize`.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='ScaleAspectJitter',
  img_scale=None,
  keep_ratio=False,
  resize_type='indep_sample_in_range',
  scale_range=(640, 2560))
```

</td><td>

```python
 dict(
  type='RandomResize',
  scale=(640, 640),
  ratio_range=(1.0, 4.125),
  resize_type='Resize',
  keep_ratio=True))
```

</td></tr>
</thead>
</table>

When `resize_type='long_short_bound'`, we implemented `BoundedScaleAspectJitter`, which randomly rescales the image so that the long and short sides of the image are around the bound; then jitter the aspect ratio.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='ScaleAspectJitter',
  img_scale=[(3000, 736)],  # Unused
  ratio_range=(0.7, 1.3),
  aspect_ratio_range=(0.9, 1.1),
  multiscale_mode='value',
  long_size_bound=800,
  short_size_bound=480,
  resize_type='long_short_bound',
  keep_ratio=False)
```

</td><td>

```python
dict(
  type='BoundedScaleAspectJitter',
  long_size_bound=800,
  short_size_bound=480,
  ratio_range=(0.7, 1.3),
  aspect_ratio_range=(0.9, 1.1))
```

</td></tr>
</thead>
</table>

When `resize_type='round_min_img_scale'`, we implemented `ShortScaleAspectJitter`, which firstly rescales the image for its shorter side to reach the `short_size` and then jitters its aspect ratio, final rescale the shape guaranteed to be divided by scale_divisor.

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x Configuration</th>
    <th>MMOCR 1.x Configuration</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='ScaleAspectJitter',
  img_scale=[(3000, 640)],
  ratio_range=(0.7, 1.3),
  aspect_ratio_range=(0.9, 1.1),
  multiscale_mode='value',
  keep_ratio=False)
```

</td><td>

```python
dict(
  type='ShortScaleAspectJitter',
  short_size=640,
  ratio_range=(0.7, 1.3),
  aspect_ratio_range=(0.9, 1.1),
  scale_divisor=32),
```

</td></tr>
</thead>
</table>
