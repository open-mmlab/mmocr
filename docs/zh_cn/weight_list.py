from ..en.weight_list import (en_KIEWeightList, en_TextDetWeightList,
                              en_TextRecWeightList)


class TextDetWeightList(en_TextDetWeightList):

    msg = '### 文字检测'


class TextRecWeightList(en_TextRecWeightList):

    msg = ('### 文字识别\n'
           '```{note}\n'
           'Avg 指该模型在 IIIT5K、SVT、ICDAR2013、ICDAR2015、SVTP、'
           'CT80 上的平均结果。\n```\n')


class KIEWeightList(en_KIEWeightList):

    msg = '### 关键信息提取'


def gen_weight_list():
    content = TextDetWeightList().gen_model_list()
    content += TextRecWeightList().gen_model_list()
    content += KIEWeightList().gen_model_list()
    return content
