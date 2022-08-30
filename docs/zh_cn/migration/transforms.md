# 数据变换迁移

## 简介

MMOCR 0.x 版本中，我们在 `mmocr/datasets/pipelines/xxx_transforms.py` 中实现了一系列的数据变换（Data Transforms）方法。然而，这些模块分散在各处，且缺乏规范统一的设计。因此，我们在 MMOCR 1.x 版本中对所有的数据增强模块进行了重构，并依照任务类型分别存放在 `mmocr/datasets/transforms` 目录下的 `ocr_transforms.py`，`textdet_transforms.py` 及 `textrecog_transforms.py` 中。其中，`ocr_transforms.py` 中实现了 OCR 相关任务通用的数据增强模块，而 `textdet_transforms.py` 和 `textrecog_transforms.py` 则分别实现了文本检测任务与文本识别任务相关的数据增强模组。

由于在重构过程中我们对部分模块进行了重命名、合并或拆分，使得新的调用接口与默认参数可能与旧版本存在不一致。因此，本文档将详细介绍如何对数据增强模块进行迁移，即，如何配置现有的数据变换来达到与旧版一致的行为。

## 配置迁移指南

### 数据格式化相关数据变换

1. `Collect` + `CustomFormatBundle` -> [`PackTextDetInputs`](mmocr.datasets.transforms.PackTextDetInputs)/[`PackTextRecogInputs`](mmocr.datasets.transforms.PackTextRecogInputs)

`PackxxxInputs` 同时囊括了 `Collect` 和 `CustomFormatBundle` 两个功能，且不再有 `key` 参数，而训练目标 target 的生成现在被转移至在 `loss` 中完成。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
    type='CustomFormatBundle',
    keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
    meta_keys=['img_path', 'ori_shape', 'img_shape'],
    visualize=dict(flag=False, boundary_key='gt_shrink')),
dict(
    type='Collect',
    keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
```

</td><td>

```python
dict(
  type='PackTextDetInputs',
  meta_keys=('img_path', 'ori_shape', 'img_shape'))
```

</td></tr>
</thead>
</table>

### 数据增强相关数据变换

1. `ResizeOCR` -> [`Resize`](mmocr.datasets.transforms.Resize), [`RescaleToHeight`](mmocr.datasets.transforms.RescaleToHeight), [`PadToWidth`](mmocr.datasets.transforms.PadToWidth)

   原有的 `ResizeOCR` 现在被拆分为三个独立的数据增强模块。

   `keep_aspect_ratio=False` 时，等价为 1.x 版本中的 `Resize`，其配置可按如下方式修改。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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

`keep_aspect_ratio=True`，且 `max_width=None` 时。将图片的高缩放至固定值，并等比例缩放图像的宽。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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

`keep_aspect_ratio=True`，且 `max_width` 为固定值时。将图片的高缩放至固定值，并等比例缩放图像的宽。若缩放后的图像宽小于 `max_width`, 则将其填充至 `max_width`, 反之则将其裁剪至 `max_width`。即，输出图像的尺寸固定为 `(height, max_width)`。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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

2. `RandomRotateTextDet` &  `RandomRotatePolyInstances` -> [`RandomRotate`](mmocr.datasets.transforms.RandomRotate)

   随机旋转数据增强策略已被整合至 `RanomRotate`。该方法的默认行为与 0.x 版本中的 `RandomRotateTextDet` 保持一致。此时仅需指定最大旋转角度 `max_angle` 即可。

```{note}
  新旧版本 "max_angle" 的默认值不同，因此需要重新进行指定。
```

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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

对于 `RandomRotatePolyInstances`，则需要指定参数 `use_canvas=True`。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='RandomRotatePolyInstances',
  rotate_ratio=0.5, # 指定概率为0.5
  max_angle=60,
  pad_with_fixed_color=False)
```

</td><td>

```python
# 用 RandomApply 对数据变换进行包装，并指定执行概率
dict(
  type='RandomApply',
  transforms=[
    dict(type='RandomRotate',
    max_angle=60,
    pad_with_fixed_color=False,
    use_canvas=True)],
  prob=0.5) # 设置执行概率为 0.5
```

</td></tr>
</thead>
</table>

```{note}
在 0.x 版本中，部分数据增强方法通过定义一个内部变量 'xxx_ratio' 来指定执行概率，如 'rotate_ratio', 'crop_ratio' 等。在 1.x 版本中，这些参数已被统一删除。现在，我们可以通过 'RandomApply' 来对不同的数据变换方法进行包装，并指定其执行概率。
```

3. `RandomCropFlip` -> [`TextDetRandomCropFlip`](mmocr.datasets.transforms.TextDetRandomCropFlip)

   目前仅对方法名进行了更改，其他参数保持一致。

4. `RandomCropPolyInstances` -> [`RandomCrop`](mmocr.datasets.transforms.RandomCrop)

   新版本移除了 `crop_ratio` 以及 `instance_key`，并统一使用 `gt_polygons` 为目标进行裁剪。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='RandomCropPolyInstances',
  instance_key='gt_masks',
  crop_ratio=0.8, # 指定概率为 0.8
  min_side_ratio=0.3)
```

</td><td>

```python
# 用 RandomApply 对数据变换进行包装，并指定执行概率
dict(
  type='RandomApply',
  transforms=[dict(type='RandomCrop', min_side_ratio=0.3)],
  prob=0.8) # 设置执行概率为 0.8
```

</td></tr>
</thead>
</table>

5. `RandomCropInstances` -> [`TextDetRandomCrop`](mmocr.datasets.transforms.TextDetRandomCrop)

   新版本移除了 `instance_key` 和 `mask_type`，并统一使用 `gt_polygons` 为目标进行裁剪。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
  </tr>
  <tbody><tr>
  <td valign="top">

```python
dict(
  type='RandomCropInstances',
  target_size=(800，800),
  instance_key='gt_kernels')
```

</td><td>

```python
dict(
  type='TextDetRandomCrop',
  target_size=(800，800))
```

</td></tr>
</thead>
</table>

6. `EastRandomCrop` -> [`RandomCrop`](mmocr.datasets.transforms.RandomCrop) + [`Resize`](mmocr.datasets.transforms.Resize) + [`mmcv.Pad`](mmcv.transforms.Pad)

   原有的 `EastRandomCrop` 内同时对图像进行了剪裁、缩放以及填充。在新版本中，我们可以通过组合三种数据增强策略来达到相同的效果。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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

   在新版本中，我们直接使用 mmcv 中实现的 `RandomResize` 来代替原有的实现。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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
默认地，数据流水线会从当前 *scope* 的注册器中搜索对应的数据变换，如果不存在该数据变化，则将继续在上游库，如 MMCV 中进行搜索。例如，MMOCR 中并未实现 `RandomResize` 方法，但我们仍然可以在配置中直接引用该数据增强方法，因为程序将自动从上游的 MMCV 中搜索该方法。此外，用户也可以通过添加前缀的形式来指定 *scope*。例如，`mmcv.RandomResize` 将强制指定使用 MMCV 库中实现的 `RandomResize`，当上下游库中存在同名方法时，则可以通过这种形式强制使用特定的版本。
```

8. `SquareResizePad` -> [`Resize`](mmocr.datasets.transforms.Resize) + [`SourceImagePad`](mmocr.datasets.transforms.SourceImagePad)

   原有的 `SquareResizePad` 内部实现了两个分支，并依据概率 `pad_ratio` 随机使用其中的一个分支进行数据增强。具体而言，一个分支先对图像缩放再填充；另一个分支则直接对图像进行缩放。为增强不同模块的复用性，我们在 1.x 版本中将该方法拆分成了 `Resize` + `SourceImagePad` 的组合形式，并通过 MMCV 中的 `RandomChoice` 来控制分支。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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
  prob=[0.4, 0.6]), # 两种组合的选用概率
```

</td></tr>
</thead>
</table>

```{note}
在 1.x 版本中，随机选择包装器 "RandomChoice" 代替了 "OneOfWrapper"，可以从一系列数据变换组合中随机抽取一组并应用。
```

9. `RandomWrapper` -> `mmcv.RandomApply`

   在 1.x 版本中，`RandomWrapper` 包装器被替换为由 MMCV 实现的 `RandomApply`，用以指定数据变换的执行概率。其中概率 `p` 现在被命名为 `prob`。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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

```
随机选择包装器现在被重命名为 `RandomChoice`，并且使用方法和原来完全一致。
```

11. `ScaleAspectJitter` -> [`ShortScaleAspectJitter`](mmocr.datasets.transforms.ShortScaleAspectJitter), [`BoundedScaleAspectJitter`](mmocr.datasets.transforms.BoundedScaleAspectJitter)

```
原有的 `ScaleAspectJitter` 实现了多种不同的图像尺寸抖动数据增强策略，在新版本中，我们将其拆分为数个逻辑更加清晰的独立数据变化方法。

`resize_type='indep_sample_in_range'` 时，其等价于图像在指定范围内的随机缩放。
```

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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

`resize_type='long_short_bound'` 时，将图像缩放至指定大小，再对其长宽比进行抖动。这一逻辑现在由新的数据变换类 `BoundedScaleAspectJitter` 实现。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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

`resize_type='around_min_img_scale'` （默认参数）时，将图像的短边缩放至指定大小，再在指定范围内对长宽比进行抖动。最后，确保其边长能被 `scale_divisor` 整除。这一逻辑由新的数据变换类 `ShortScaleAspectJitter` 实现。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 配置</th>
    <th>MMOCR 1.x 配置</th>
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
