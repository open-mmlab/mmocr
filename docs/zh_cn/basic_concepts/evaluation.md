# 性能评测

## 评测指标

MMOCR 基于 {external+mmengine:doc}`MMEngine: BaseMetric <design/evaluation>` 基类实现了常用的文本检测、文本识别以及关键信息抽取任务的评测指标，用户可以通过修改配置文件中的 `val_evaluator` 与 `test_evaluator` 字段来便捷地指定验证与测试阶段采用的评测方法。例如，以下配置展示了如何在文本检测算法中使用 `HmeanIOUMetric` 来评测模型性能。

```python
val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = val_evaluator
```

```{tip}
更多评测相关配置请参考评[测配置教程](../user_guides/config.md#评测配置)。
```

如下表所示，MMOCR 目前针对文本检测、识别、及关键信息抽取等任务共内置了 5 种评测指标，分别为 `HmeanIOUMetric`，`WordMetric`，`CharMetric`，`OneMinusNEDMetric`，和 `F1Metric`。

|                                                                        |              |                                                   |                                                                       |
| ---------------------------------------------------------------------- | ------------ | ------------------------------------------------- | --------------------------------------------------------------------- |
| 评测指标                                                               | 任务类型     | 输入字段                                          | 输出字段                                                              |
| [HmeanIOUMetric](mmocr.evaluation.metrics.hmean_iou_metric.HmeanIOUMetric) | 文本检测     | `pred_polygons`<br>`pred_scores`<br>`gt_polygons` | `recall`<br>`precision`<br>`hmean`                                    |
| [WordMetric](mmocr.evaluation.metrics.recog_metric.WordMetric)         | 文本识别     | `pred_text`<br>`gt_text`                          | `word_acc`<br>`word_acc_ignore_case`<br>`word_acc_ignore_case_symbol` |
| [CharMetric](mmocr.evaluation.metrics.recog_metric.CharMetric)         | 文本识别     | `pred_text`<br>`gt_text`                          | `char_recall`<br>`char_precision`                                     |
| [OneMinusNEDMetric](mmocr.evaluation.metrics.recog_metric.OneMinusNEDMetric) | 文本识别     | `pred_text`<br>`gt_text`                          | `1-N.E.D`                                                             |
| [F1Metric](mmocr.evaluation.metrics.f_metric.F1Metric)                 | 关键信息抽取 | `pred_labels`<br>`gt_labels`                      | `macro_f1`<br>`micro_f1`                                              |

通常来说，每一类任务所采用的评测标准是约定俗成的，用户一般无须深入了解或手动修改评测方法的内部实现。然而，为了方便用户实现更加定制化的需求，本文档将进一步介绍了 MMOCR 内置评测算法的具体实现策略，以及可配置参数。

### HmeanIOUMetric

`HmeanIOUMetric` 是文本检测任务中应用最广泛的评测指标之一，因其计算了检测精度（Precision）与召回率（Recall）之间的调和平均数（Harmonic mean, H-mean），故得名 `HmeanIOUMetric`。记精度为 *P*，召回率为 *R*，则 `HmeanIOUMetric` 可由下式计算得到：

```{math}
H = \frac{2}{\frac{1}{P} + \frac{1}{R}} = \frac{2PR}{P+R}
```

另外，由于其等价于 {math}`\beta = 1` 时的 F-score (又称 F-measure 或 F-metric)，`HmeanIOUMetric` 有时也被写作 `F1Metric` 或 `f1-score` 等：

```{math}
F_1=(1+\beta^2)\cdot\frac{PR}{\beta^2\cdot P+R} = \frac{2PR}{P+R}
```

在 MMOCR 的设计中，`HmeanIOUMetric` 的计算可以概括为以下几个步骤：

1. 过滤无效的预测边界盒

   - 依据置信度阈值 `pred_score_thrs` 过滤掉得分较低的预测边界盒
   - 依据 `ignore_precision_thr` 阈值过滤掉与 `ignored` 样本重合度过高的预测边界盒

   值得注意的是，`pred_score_thrs` 默认将**自动搜索**一定范围内的**最佳阈值**，用户也可以通过手动修改配置文件来自定义搜索范围：

   ```python
   # HmeanIOUMetric 默认以 0.1 为步长搜索 [0.3, 0.9] 范围内的最佳得分阈值
   val_evaluator = dict(type='HmeanIOUMetric', pred_score_thrs=dict(start=0.3, stop=0.9, step=0.1))
   ```

2. 计算 IoU 矩阵

   - 在数据处理阶段，`HmeanIOUMetric` 会计算并维护一个 {math}`M \times N` 的 IoU 矩阵 `iou_metric`，以方便后续的边界盒配对步骤。其中，M 和 N 分别为标签边界盒与预测边界盒的数量。由此，该矩阵的每个元素都存放了第 m 个标签边界盒与第 n 个预测边界盒之间的交并比（IoU）。

3. 基于相应的配对策略统计能被准确匹配的 GT 样本数

   尽管 `HmeanIOUMetric` 可以由固定的公式计算取得，不同的任务或算法库内部的具体实现仍可能存在一些细微差别。这些差异主要体现在采用不同的策略来匹配真实与预测边界盒，从而导致最终得分的差距。目前，MMOCR 内部的 `HmeanIOUMetric` 共支持两种不同的匹配策略，即 `vanilla` 与 `max_matching`。如下所示，用户可以通过修改配置文件来指定不同的匹配策略。

   - `vanilla` 匹配策略

     `HmeanIOUMetric` 默认采用 `vanilla` 匹配策略，该实现与 MMOCR 0.x 版本中的 `hmean-iou` 及 ICDAR 系列**官方文本检测竞赛的评测标准保持一致**，采用先到先得的匹配方式对标签边界盒（Ground-truth bbox）与预测边界盒（Predicted bbox）进行配对。

     ```python
     # 不指定 strategy 时，HmeanIOUMetric 默认采用 'vanilla' 匹配策略
     val_evaluator = dict(type='HmeanIOUMetric')
     ```

   - `max_matching` 匹配策略

     针对现有匹配机制中的不完善之处，MMOCR 算法库实现了一套更高效的匹配策略，用以最大化匹配数目。

     ```python
     # 指定采用 'max_matching' 匹配策略
     val_evaluator = dict(type='HmeanIOUMetric', strategy='max_matching')
     ```

   ```{note}
   我们建议面向学术研究的开发用户采用默认的 `vanilla` 匹配策略，以保证与其他论文的对比结果保持一致。而面向工业应用的开发用户则可以采用 `max_matching` 匹配策略，以获得更高的性能。
   ```

4. 根据上文介绍的 `HmeanIOUMetric` 公式计算最终的评测得分

### WordMetric

`WordMetric` 实现了**单词级别**的文本识别评测指标，并内置了 `exact`，`ignore_case`，及 `ignore_case_symbol` 三种文本匹配模式，用户可以在配置文件中修改 `mode` 字段来自由组合输出一种或多种文本匹配模式下的 `WordMetric` 得分。

```python
# 在文本识别任务中使用 WordMetric 评测
val_evaluator = [
    dict(type='WordMetric', mode=['exact', 'ignore_case', 'ignore_case_symbol'])
]
```

- `exact`：全匹配模式，即，预测与标签完全一致才能被记录为正确样本。
- `ignore_case`：忽略大小写的匹配模式。
- `ignore_case_symbol`：忽略大小写及符号的匹配模式，这也是大部分学术论文中报告的文本识别准确率；MMOCR 报告的识别模型性能默认采用该匹配模式。

假设真实标签为 `MMOCR!`，模型的输出结果为 `mmocr`，则三种匹配模式下的 `WordMetric` 得分分别为：`{'exact': 0, 'ignore_case': 0, 'ignore_case_symbol': 1}`。

### CharMetric

`CharMetric` 实现了**不区分大小写**的**字符级别**的文本识别评测指标。

```python
# 在文本识别任务中使用 CharMetric 评测
val_evaluator = [dict(type='CharMetric')]

# 此外，MMOCR 也支持相同任务下的多种指标组合评测，如同时使用 WordMetric 及 CharMetric
val_evaluator = [
    dict(type='WordMetric', mode=['exact', 'ignore_case', 'ignore_case_symbol']),
    dict(type='CharMetric')
]
```

具体而言，`CharMetric` 会输出两个评测评测指标，即字符精度 `char_precision` 和字符召回率 `char_recall`。设正确预测的字符（True Positive）数量为 {math}`\sigma_{tp}`，则精度 *P* 和召回率 *R* 可由下式计算取得：

```{math}
P=\frac{\sigma_{tp}}{\sigma_{gt}}, R = \frac{\sigma_{tp}}{\sigma_{pred}}
```

其中，{math}`\sigma_{gt}` 与 {math}`\sigma_{pred}` 分别为标签文本与预测文本所包含的字符总数。

例如，假设标签文本为 "MM**O**CR"，预测文本为 "mm**0**cR**1**"，则使用 `CharMetric` 评测指标的得分为：

```{math}
P=\frac{4}{5}, R=\frac{4}{6}
```

### OneMinusNEDMetric

`OneMinusNEDMetric（1-N.E.D）` 常用于中文或英文**文本行级别**标注的文本识别评测，不同于全匹配的评测标准要求预测与真实样本完全一致，该评测指标使用归一化的[编辑距离](https://en.wikipedia.org/wiki/Edit_distance)（Edit Distance，又名莱温斯坦距离 Levenshtein Distance）来测量预测文本与真实文本之间的差异性，从而在评测长文本样本时能够更好地区分出模型的性能差异。假设真实和预测文本分别为 {math}`s_i` 和 {math}`\hat{s_i}`，其长度分别为 {math}`l_{i}` 和 {math}`\hat{l_i}`，则 `OneMinusNEDMetric` 得分可由下式计算得到：

```{math}
score = 1 - \frac{1}{N}\sum_{i=1}^{N}\frac{D(s_i, \hat{s_{i}})}{max(l_{i},\hat{l_{i}})}
```

其中，*N* 是样本总数，{math}`D(s_1, s_2)` 为两个字符串之间的编辑距离。

例如，假设真实标签为 "OpenMMLabMMOCR"，模型 A 的预测结果为 "0penMMLabMMOCR", 模型 B 的预测结果为 "uvwxyz"，则采用全匹配和 `OneMinusNEDMetric` 评测指标的结果分别为:

|        |        |            |
| ------ | ------ | ---------- |
|        | 全匹配 | 1 - N.E.D. |
| 模型 A | 0      | 0.92857    |
| 模型 B | 0      | 0          |

由上表可以发现，尽管模型 A 仅预测错了一个字母，而模型 B 全部预测错误，在使用全匹配的评测指标时，这两个模型的得分都为0；而使用 `OneMinuesNEDMetric` 的评测指标则能够更好地区分模型在**长文本**上的性能差异。

### F1Metric

`F1Metric` 实现了针对 KIE 任务的 F1-Metric 评测指标，并提供了 `micro` 和 `macro` 两种评测模式。

```python
val_evaluator = [
    dict(type='F1Metric', mode=['micro', 'macro'],
]
```

- `micro` 模式：依据 True Positive，False Negative，及 False Positive 总数来计算全局 F1-Metric 得分。

- `macro` 模式：依据类别标签计算每一类的 F1-Metric，并求平均值。

### 自定义评测指标

对于追求更高定制化功能的用户，MMOCR 也支持自定义实现不同类型的评测指标。一般来说，用户只需要新建自定义评测指标类 `CustomizedMetric` 并继承 {external+mmengine:doc}`MMEngine: BaseMetric <design/evaluation>`，然后分别重写数据格式处理方法 `process` 以及指标计算方法 `compute_metrics`。最后，将其加入 `METRICS` 注册器即可实现任意定制化的评测指标。

```python
from mmengine.evaluator import BaseMetric
from mmocr.registry import METRICS

@METRICS.register_module()
class CustomizedMetric(BaseMetric):

    def process(self, data_batch: Sequence[Dict], predictions: Sequence[Dict]):
    """ process 接收两个参数，分别为 data_batch 存放真实标签信息，以及 predictions
        存放预测结果。process 方法负责将标签信息转换并存放至 self.results 变量中
    """
        pass

    def compute_metrics(self, results: List):
    """ compute_metric 使用经过 process 方法处理过的标签数据计算最终评测得分
    """
        pass
```

```{note}
更多内容可参见 {external+mmengine:doc}`MMEngine 文档: BaseMetric <design/evaluation>`。
```
