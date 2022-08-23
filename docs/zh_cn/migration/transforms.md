# 数据变换（Data Transforms）迁移

## 简介

MMOCR 0.x 版本中，我们在 `mmocr/datasets/pipelines/xxx_transforms.py` 中实现了一系列的数据变换与数据增强方法。然而，这些模块分散在各处，且缺乏规范统一的设计。因此，我们在 MMOCR 1.x 版本中对所有的数据增强模块进行了重构，并依照其任务类型分别存放在 `mmocr/datasets/transforms` 目录下的 `ocr_transforms.py`，`textdet_transforms.py` 及 `textrecog_transforms.py` 中。其中，`ocr_transforms.py` 中实现了 OCR 相关任务通用的数据增强模块，而 `textdet_transforms.py` 和 `textrecog_transforms.py` 则分别实现了文本检测任务与文本识别任务相关的数据增强模组。

由于在重构过程中我们对部分模块进行了重命名、合并或拆分，使得新的调用接口与默认参数可能与旧版本存在不一致。因此，本文档将详细介绍如何对数据增强模块进行迁移，即，如何配置现有的数据变换来达到与旧版一致的行为。

## 配置迁移指南

### 数据格式化相关数据变换

1. `Collect` + `CustomFormatBundle` -> `PackTextDetInputs/PackTextRecogInputs`

`PackxxxInputs` 同时囊括了 `Collect` 和 `CustomFormatBundle` 两个功能，且不再有 `key` 参数，训练目标 target 的生成被转移至在 `loss` 中完成。

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

### 数据增强相关数据变换

1. `ResizeOCR` -> `Resize`, `RescaleToHeight`, `PadToWidth`

原有的 `ResizeOCR` 现在被拆分为三个独立的数据增强模块。

`keep_aspect_ratio=False` 时，等价为 1.x 版本中的 Resize，其配置可按如下方式修改。

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

`keep_aspect_ratio=True`，且 `max_width` 为固定值时。将图片的高缩放至固定值，并等比例缩放图像的宽。若缩放后的图像宽小于 `max_width`, 则将其 `padding` 至 `max_width`, 反之则将其裁剪至 `max_width`。即，输出图像的尺寸固定为 `(height, max_width)`。

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

2. `RandomRotateTextDet` &  `RandomRotatePolyInstances` -> `RandomRotate`

随机旋转数据增强策略已被整合至 `RanomRotate`。该方法的默认行为与 0.x 版本中的 `RandomRotateTextDet` 保持一致。此时仅需指定最大旋转角度 `max_angle` 即可。

```{note}
  新旧版本 'max_angle' 的默认值不同，因此需要重新进行指定。
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
  rotate_ratio=0.5,
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
在 0.x 版本中，部分数据变化通过定义一个内部变量来指定执行概率，如 'rotate_ratio' 等。在 1.x 版本中，这些参数已被统一删除。现在，我们可以通过 'RandomApply' 来对不同的数据变换方法进行包装，并指定其执行概率。
```
