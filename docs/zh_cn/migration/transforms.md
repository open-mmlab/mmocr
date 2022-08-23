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
    <th>MMOCR 0.x 旧版配置</th>
    <th>MMOCR 1.x 新版配置</th>
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

   原有的 `ResizeOCR` 现在被拆分为三个独立数据增强模块。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 旧版配置</th>
    <th>MMOCR 1.x 新版配置</th>
  </tr>
  <tbody>
  <tr><td valign="top">

```python
# keep_aspect_ratio=False 时，等价为
# 1.x 版本中的 Resize
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
<tr><td>

```python
# keep_aspect_ratio=True，且
# max_width=None 时，将图片的
# 高缩放至固定值，并等比例缩放图像的宽
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
<tr><td>

```python
# keep_aspect_ratio=True，且
# max_width 为固定值时，将图片的
# 高缩放至固定值，并等比例缩放图像的宽。
# 若缩放后的图像宽小于 max_width,
# 则 padding 至 max_width, 反之
# 则 crop 至 max_width。即，输出
# 图像的尺寸固定为 (height, max_width)。
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
