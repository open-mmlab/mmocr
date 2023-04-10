import os.path as osp

from mmengine.fileio import load
from tabulate import tabulate


class BaseWeightList:
    """Class for generating model list in markdown format.

    Args:
        dataset_list (list[str]): List of dataset names.
        table_header (list[str]): List of table header.
        msg (str): Message to be displayed.
        task_abbr (str): Abbreviation of task name.
        metric_name (str): Metric name.
    """

    base_url: str = 'https://github.com/open-mmlab/mmocr/blob/1.x/'
    table_cfg: dict = dict(
        tablefmt='pipe', floatfmt='.2f', numalign='right', stralign='center')
    dataset_list: list
    table_header: list
    msg: str
    task_abbr: str
    metric_name: str

    def __init__(self):
        data = (d + f' ({self.metric_name})' for d in self.dataset_list)
        self.table_header = ['模型', 'README', *data]

    def _get_model_info(self, task_name: str):
        meta_indexes = load('../../model-index.yml')
        for meta_path in meta_indexes['Import']:
            meta_path = osp.join('../../', meta_path)
            metainfo = load(meta_path)
            collection2md = {}
            for item in metainfo['Collections']:
                url = self.base_url + item['README']
                collection2md[item['Name']] = f'[链接]({url})'
            for item in metainfo['Models']:
                if task_name not in item['Config']:
                    continue
                name = f'`{item["Name"]}`'
                if item.get('Alias', None):
                    if isinstance(item['Alias'], str):
                        item['Alias'] = [item['Alias']]
                    aliases = [f'`{alias}`' for alias in item['Alias']]
                    aliases.append(name)
                    name = ' / '.join(aliases)
                readme = collection2md[item['In Collection']]
                eval_res = self._get_eval_res(item)
                yield (name, readme, *eval_res)

    def _get_eval_res(self, item):
        eval_res = {k: '-' for k in self.dataset_list}
        for res in item['Results']:
            if res['Dataset'] in self.dataset_list:
                eval_res[res['Dataset']] = res['Metrics'][self.metric_name]
        return (eval_res[k] for k in self.dataset_list)

    def gen_model_list(self):
        content = f'\n{self.msg}\n'
        content += '```{table}\n:class: model-summary nowrap field-list '
        content += 'table table-hover\n'
        content += tabulate(
            self._get_model_info(self.task_abbr), self.table_header,
            **self.table_cfg)
        content += '\n```\n'
        return content


class TextDetWeightList(BaseWeightList):

    dataset_list = ['ICDAR2015', 'CTW1500', 'Totaltext']
    msg = '### 文字检测'
    task_abbr = 'textdet'
    metric_name = 'hmean-iou'


class TextRecWeightList(BaseWeightList):

    dataset_list = [
        'Avg', 'IIIT5K', 'SVT', 'ICDAR2013', 'ICDAR2015', 'SVTP', 'CT80'
    ]
    msg = ('### 文字识别\n'
           '```{note}\n'
           'Avg 指该模型在 IIIT5K、SVT、ICDAR2013、ICDAR2015、SVTP、'
           'CT80 上的平均结果。\n```\n')
    task_abbr = 'textrecog'
    metric_name = 'word_acc'

    def _get_eval_res(self, item):
        eval_res = {k: '-' for k in self.dataset_list}
        avg = []
        for res in item['Results']:
            if res['Dataset'] in self.dataset_list:
                eval_res[res['Dataset']] = res['Metrics'][self.metric_name]
                avg.append(res['Metrics'][self.metric_name])
        eval_res['Avg'] = sum(avg) / len(avg)
        return (eval_res[k] for k in self.dataset_list)


class KIEWeightList(BaseWeightList):

    dataset_list = ['wildreceipt']
    task_abbr = 'kie'
    metric_name = 'macro_f1'
    msg = '### 关键信息提取'


def gen_weight_list():
    content = TextDetWeightList().gen_model_list()
    content += TextRecWeightList().gen_model_list()
    content += KIEWeightList().gen_model_list()
    return content
