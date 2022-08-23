# 数据增强模块迁移

MMOCR 0.x 版本中，我们在 `mmocr/datasets/pipelines/xxx_transforms.py` 中实现了一系列的数据变换与数据增强方法。然而，这些模块分散在各处，且缺乏规范统一的设计。因此，我们在 MMOCR 1.x 版本中对所有的数据增强模块进行了重构，并依照其任务类型分别存放在 `mmocr/datasets/transforms` 目录下的 `ocr_transforms.py`，`textdet_transforms.py` 及 `textrecog_transforms.py` 中。其中，`ocr_transforms.py` 中实现了 OCR 相关任务通用的数据增强模块，而 `textdet_transforms.py` 和 `textrecog_transforms.py` 则分别实现了文本检测任务与文本识别任务相关的数据增强模组。

由于在重构过程中我们对部分模块进行了合并或重名，使得新的调用接口与默认参数可能与旧版本存在不一致。因此，本文档将详细介绍如何对数据增强模块进行迁移，即，如何配置现有的数据变换来达到与旧版一致的行为。

## 配置迁移说明

1. `Collect` + `CustomFormatBundle` -> `PackTextDetInputs/PackTextRecogInputs`

`PackxxxInputs` 同时囊括了 `Collect` 和 `CustomFormatBundle` 两个功能，且不再有 `key` 参数，训练目标 target 的生成被转移至在 `loss` 中完成。

<table class="docutils">
<thead>
  <tr>
    <th>MMOCR 0.x 旧版配置</th>
    <th>MMOCR 1.x 新版配置</th>
  <tbody><tr>
  <td valign="top">

```python
  dict(
      type='CustomFormatBundle',
      keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
      meta_keys=['img_path', 'ori_shape', 'img_shape',
                          'scale_factor', 'flip', 'flip_direction']
      visualize=dict(flag=False, boundary_key='gt_shrink')),v
  dict(
      type='Collect',
      keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
```

</td><td>

```python
  dict(type='PackTextDetInputs', meta_keys=('img_path', 'ori_shape',
                                           'img_shape', 'scale_factor',
                                           'flip', 'flip_direction'))
```

</td></tr>
</thead>
</table>
