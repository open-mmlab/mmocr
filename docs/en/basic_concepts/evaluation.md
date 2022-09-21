# Evaluation

## Metrics

MMOCR implements widely-used evaluation metrics for text detection, text recognition and key information extraction tasks based on the {external+mmengine:doc}`MMEngine: BaseMetric <design/evaluation>` base class. Users can specify the metric used in the validation and test phases by modifying the `val_evaluator` and `test_evaluator` fields in the configuration file. For example, the following config shows how to use `HmeanIOUMetric` to evaluate the model performance in text detection task.

```python
val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = val_evaluator
```

```{tips}
More evaluation related configurations can be found in the [evaluation configuration tutorial](../user_guides/config.md#evaluation-configuration).
```

As shown in the following table, MMOCR currently supports 5 evaluation metrics for text detection, text recognition, and key information extraction tasks, including `HmeanIOUMetric`, `WordMetric`, `CharMetric`, `OneMinusNEDMetric`, and `F1Metric`.

|                                                                 |         |                                                   |                                                                       |
| --------------------------------------------------------------- | ------- | ------------------------------------------------- | --------------------------------------------------------------------- |
| Metric                                                          | Task    | Input Field                                       | Output Field                                                          |
| [HmeanIOUMetric](mmocr.evaluation.metrics.HmeanIOUMetric)       | TextDet | `pred_polygons`<br>`pred_scores`<br>`gt_polygons` | `recall`<br>`precision`<br>`hmean`                                    |
| [WordMetric](mmocr.evaluation.metrics.WordMetric)               | TextRec | `pred_text`<br>`gt_text`                          | `word_acc`<br>`word_acc_ignore_case`<br>`word_acc_ignore_case_symbol` |
| [CharMetric](mmocr.evaluation.metrics.CharMetric)               | TextRec | `pred_text`<br>`gt_text`                          | `char_recall`<br>`char_precision`                                     |
| [OneMinusNEDMetric](mmocr.evaluation.metrics.OneMinusNEDMetric) | TextRec | `pred_text`<br>`gt_text`                          | `1-N.E.D`                                                             |
| [F1Metric](mmocr.evaluation.metrics.F1Metric)                   | KIE     | `pred_labels`<br>`gt_labels`                      | `macro_f1`<br>`micro_f1`                                              |

In general, the evaluation metric used in each task is conventionally determined. Users usually do not need to understand or manually modify the internal implementation of the evaluation metric. However, to facilitate more customized requirements, this document will further introduce the specific implementation details and configurable parameters of the built-in metrics in MMOCR.

### HmeanIOUMetric

`HmeanIOUMetric` is one of the most widely used evaluation metrics in text detection tasks, because it calculates the harmonic mean (H-mean) between the detection precision (P) and recall rate (R). The `HmeanIOUMetric` can be calculated by the following equation:

$$H=\\frac{2}{\\frac{1}{P}+\\frac{1}{R}}=\\frac{2PR}{P+R}$$

In addition, since it is equivalent to the F-score (also known as F-measure or F-metric) when $$\\beta = 1$$, `HmeanIOUMetric` is sometimes written as `F1Metric` or `f1-score`:

$$F_1=(1+\\beta^2)\\cdot\\frac{PR}{\\beta^2\\cdot P+R} = \\frac{2PR}{P+R}$$

In MMOCR, the calculation of `HmeanIOUMetric` can be summarized as the following steps:

1. Filter out invalid predictions

   - Filter out predictions with a score lower than `pred_score_thrs`
   - Filter out predictions overlapping with `ignored` ground truth boxes with an overlap ratio higher than `ignore_precision_thr`

   It is worth noting that `pred_score_thrs` will **automatically search** for the **best threshold** within a certain range by default, and users can also customize the search range by manually modifying the configuration file:

   ```python
   # By default, HmeanIOUMetric searches the best threshold within the range [0.3, 0.9] with a step size of 0.1
   val_evaluator = dict(type='HmeanIOUMetric', pred_score_thrs=dict(start=0.3, stop=0.9, step=0.1))
   ```

2. Calculate the IoU matrix

   - At the data processing stage, `HmeanIOUMetric` will calculate and maintain an $$M \\times N$$ IoU matrix `iou_metric` for the convenience of the subsequent bounding box pairing step. Here, M and N represent the number of label bounding boxes and prediction bounding boxes, respectively. Therefore, each element of this matrix stores the IoU between the m-th label bounding box and the n-th prediction bounding box.

3. Compute the number of GT samples that can be accurately matched based on the corresponding pairing strategy

   Although `HmeanIOUMetric` can be calculated by a fixed formula, there may still be some subtle differences in the specific implementations. These differences mainly reflect the use of different strategies to match gt and predicted bounding boxes, which leads to the difference in final scores. Currently, MMOCR supports two matching strategies, namely `vanilla` and `max_matching`, for the `HmeanIOUMetric`. As shown below, users can specify the matching strategies in the config.

   - `vanilla` matching strategy

     By default, `HmeanIOUMetric` adopts the `vanilla` matching strategy, which is consistent with the `hmean-iou` implementation in MMOCR 0.x and the **official** text detection competition evaluation standard of ICDAR series. The matching strategy adopts the first-come-first-served matching method to pair the labels and predictions.

     ```python
     # By default, HmeanIOUMetric adopts 'vanilla' matching strategy
     val_evaluator = dict(type='HmeanIOUMetric')
     ```

   - `max_matching` matching strategy

     To address the shortcomings of the existing matching mechanism, MMOCR has implemented a more efficient matching strategy to maximize the number of matches.

     ```python
     # Specify to use 'max_matching' matching strategy
     val_evaluator = dict(type='HmeanIOUMetric', strategy='max_matching')
     ```

   ```{note}
   We recommend that research-oriented developers use the default `vanilla` matching strategy to ensure consistency with other papers. For industry-oriented developers, you can use the `max_matching` matching strategy to achieve optimized performance.
   ```

4. Compute the final evaluation score according to the aforementioned matching strategy

### WordMetric

`WordMetric` implements **word-level** text recognition evaluation metrics and includes three text matching modes, namely `exact`, `ignore_case`, and `ignore_case_symbol`. Users can freely combine the output of one or more text matching modes in the configuration file by modifying the `mode` field.

```python
# Use WordMetric for text recognition task
val_evaluator = [
    dict(type='WordMetric', mode=['exact', 'ignore_case', 'ignore_case_symbol'])
]
```

- `exact`：Full matching mode, i.e., only when the predicted text and the ground truth text are exactly the same, the predicted text is considered to be correct.
- `ignore_case`：The mode ignores the case of the predicted text and the ground truth text.
- `ignore_case_symbol`：The mode ignores the case and symbols of the predicted text and the ground truth text. This is also the text recognition accuracy reported by most academic papers. The performance reported by MMOCR uses the `ignore_case_symbol` mode by default.

Assume that the real label is `MMOCR!` and the model output is `mmocr`. The `WordMetric` scores under the three matching modes are: `{'exact': 0, 'ignore_case': 0, 'ignore_case_symbol': 1}`.

### CharMetric

`CharMetric` implements **character-level** text recognition evaluation metrics that are **case-insensitive**.

```python
# Use CharMetric for text recognition task
val_evaluator = [dict(type='CharMetric')]

# In addition, MMOCR also supports the combination evaluation of multiple metrics for the same task, such as using WordMetric and CharMetric at the same time
val_evaluator = [
    dict(type='WordMetric', mode=['exact', 'ignore_case', 'ignore_case_symbol']),
    dict(type='CharMetric')
]
```

Specifically, `CharMetric` will output two evaluation metrics, namely `char_precision` and `char_recall`. Let the number of correctly predicted characters (True Positive) be $$\\sigma\_{tp}$$, then the precision *P* and recall *R* can be calculated by the following equation:

$$P=\\frac{\\sigma\_{tp}}{\\sigma\_{gt}}, R = \\frac{\\sigma\_{tp}}{\\sigma\_{pred}}$$

where $$\\sigma\_{gt}$$ and $$\\sigma\_{pred}$$ represent the total number of characters in the label text and the predicted text, respectively.

For example, assume that the label text is "MM**O**CR" and the predicted text is "mm**0**cR**1**". The score of the `CharMetric` is:

$$P=\\frac{4}{5}, R=\\frac{4}{6}$$

### OneMinusNEDMetric

`OneMinusNEDMetric（1-N.E.D）` is commonly used for text recognition evaluation of Chinese or English **text line-level** annotations. Unlike the full matching metric that requires the prediction and the gt text to be exactly the same, `1-N.E.D` uses the normalized [editing distance](https://en.wikipedia.org/wiki/Edit_distance) (also known as Levenshtein Distance) to measure the difference between the predicted and the gt text, so that the performance difference of the model can be better distinguished when evaluating long texts. Assume that the real and predicted texts are $$s_i$$ and $$\\hat{s_i}$$, respectively, and their lengths are $$l\_{i}$$ and $$\\hat{l_i}$$, respectively. The `OneMinusNEDMetric` score can be calculated by the following formula:

$$score = 1 - \\frac{1}{N}\\sum\_{i=1}^{N}\\frac{D(s_i, \\hat{s\_{i}})}{max(l\_{i},\\hat{l\_{i}})}$$

where *N* is the total number of samples, and $$D(s_1, s_2)$$ is the \[editing distance\] between two strings.

For example, assume that the real label is "OpenMMLabMMOCR", the prediction of model A is "0penMMLabMMOCR", and the prediction of model B is "uvwxyz". The results of the full matching and `OneMinusNEDMetric` evaluation metrics are as follows:

|         |            |            |
| ------- | ---------- | ---------- |
|         | Full-match | 1 - N.E.D. |
| Model A | 0          | 0.92857    |
| Model B | 0          | 0          |

As shown in the table above, although the model A only predicted one letter incorrectly, both models got 0 in when using full-match strategy. However, the `OneMinusNEDMetric` evaluation metric can better distinguish the performance of the two models on **long texts**.

### F1Metric

`F1Metric` implements the F1-Metric evaluation metric for KIE tasks and provides two modes, namely `micro` and `macro`.

```python
val_evaluator = [
    dict(type='F1Metric', mode=['micro', 'macro'],
]
```

- `micro` mode: Calculate the global F1-Metric score based on the total number of True Positive, False Negative, and False Positive.

- `macro` mode：Calculate the F1-Metric score for each class and then take the average.

### Customized Metric

MMOCR supports the implementation of customized evaluation metrics for users who pursue higher customization. In general, users only need to create a customized evaluation metric class `CustomizedMetric` and inherit {external+mmengine:doc}`MMEngine: BaseMetric <design/evaluation>`. Then, the data format processing method `process` and the metric calculation method `compute_metrics` need to be overwritten respectively. Finally, add it to the `METRICS` registry to implement any customized evaluation metric.

```python
from mmengine.evaluator import BaseMetric
from mmocr.registry import METRICS

@METRICS.register_module()
class CustomizedMetric(BaseMetric):

    def process(self, data_batch: Sequence[Dict], predictions: Sequence[Dict]):
    """ process receives two parameters, data_batch stores the gt label information, and predictions stores the predicted results.
    """
        pass

    def compute_metrics(self, results: List):
    """ compute_metric receives the results of the process method as input and returns the evaluation results.
    """
        pass
```

```{note}
More details can be found in {external+mmengine:doc}`MMEngine Documentation: BaseMetric <design/evaluation>`.
```
