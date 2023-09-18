#!/usr/bin/env python
import os
import os.path as osp

import yaml
from tabulate import tabulate


class BaseAlgorithmInfo:

    def __init__(self, info) -> None:
        self.info = info
        self.set_base_info()
        self.set_model_info()
        self.set_exp_info()

    def set_base_info(self):
        # used for overview table
        self.BaseTitle = self.get_info_value('Title')
        self.Title = self.get_info_value(
            'Title', 'get_title_format', dir='paper')
        self.Venue = self.get_info_value('Venue')
        self.Year = self.get_info_value('Year')
        self.URL = self.get_info_value('URL', method='get_paper_url_format')
        self.Institution = self.get_info_value('Lab/Company')
        self.Bibtex = self.get_info_value('Bibtex')
        self.Code = self.get_info_value('Code', method='get_link_format')

    def set_model_info(self):
        self.Abstract = self.get_info_value('Abstract')
        self.Network = self.get_info_value('Network Structure',
                                           'get_network_format')

    def set_exp_info(self):
        self.Device = self.get_info_value('Experiments.Metadata.Device')
        self.FLOPs = self.get_info_value(
            'Experiments.Metadata.FLOPs',
            'get_fps_flops_format',
            device=self.Device)
        self.Params = self.get_info_value('Experiments.Metadata.Params')
        self.TrainDataset = self.get_info_value(
            'Experiments.Metadata.Training Data')

    def get_info_value(
        self,
        key,
        method='get_item_format',
        source=None,
        return_default='N/A',
        **method_params,
    ):
        key_list = key.split('.')
        if source is None:
            source = self.info
        for k in key_list:
            if k not in source:
                return return_default
            source = source[k]
        if method is not None:
            return getattr(self, method)(source, **method_params)
        return source

    def get_fps_flops_format(self, inputs, device='N/A'):
        if inputs == 'N/A':
            return inputs
        if device == 'N/A':
            return '{:.2f}'.format(inputs)
        return '{:.2f}-{}'.format(inputs, device)

    def url_format(self, url, name='URL'):
        if url != 'N/A':
            return '[{}]({})'.format(name, url)
        return 'N/A'

    def get_paper_url_format(self, url):
        venue_url = '[Venue link]({})'.format(url['Venue']) if \
            url['Venue'] != 'N/A' else ''
        arxiv_url = '[arXiv link]({})'.format(url['Arxiv']) if \
            url['Arxiv'] != 'N/A' else ''
        return '<br>'.join([arxiv_url, venue_url]).strip('<br>')

    def get_link_format(self, url):
        return '[Link](' + url + ')' if url != 'N/A' else 'N/A'

    def get_item_format(self, item):
        if isinstance(item, list):
            return '<br>'.join(item)
        elif isinstance(item, (str, int)):
            return item

    def get_title_format(self, title, dir):
        name = title + '.md'
        name = name.replace(' ', '%20')
        return '[{}]({})'.format(title, osp.join(dir, name))

    def get_abbr_format(self, abbr, dir):
        name = self.BaseTitle + '.md'
        name = name.replace(' ', '%20')
        return '[{}]({})'.format(abbr, osp.join(dir, name))

    def get_network_format(self, network):
        return '<div align=center><img src="{}"/></div>\n\n'.format(network)

    def __getitem__(self, key):
        return getattr(self, key, 'N/A')


class RecogAlgorithmInfo(BaseAlgorithmInfo):

    def set_exp_info(self):
        super().set_exp_info()
        results = self.info['Experiments']['Results']
        self.Abbr = self.get_info_value(
            'Experiments.Name', 'get_abbr_format', dir='paper')
        self.test_dataset_list = list()
        self.metric_list = set()
        for res in results:
            self.test_dataset_list.append(res['Test Data'])
            for metric in res['Metrics']:
                self.metric_list.add(metric)
                setattr(self, metric + '_' + res['Test Data'],
                        res['Metrics'][metric])
        self.Times = self.get_info_value('Experiments.Metadata.InferenceTime')

    def set_model_info(self):
        super().set_model_info()
        self.Learning_Method = self.get_info_value('Learning Method')
        self.Language_Modality = self.get_info_value('Language Modality')
        self.Architecture = self.get_info_value('Architecture')


class DetAlgorithmInfo(BaseAlgorithmInfo):

    def set_exp_info(self):
        super().set_exp_info()
        results = self.info['Experiments']['Results']
        self.Abbr = self.get_info_value(
            'Experiments.Name', 'get_abbr_format', dir='paper')
        self.test_dataset_list = list()
        self.metric_list = set()
        for res in results:
            self.test_dataset_list.append(res['Test Data'])
            for metric in res['Metrics']:
                self.metric_list.add(metric)
                setattr(self, metric + '_' + res['Test Data'],
                        res['Metrics'][metric])
            setattr(self, 'FPS_' + res['Test Data'], res.get('FPS', 'N/A'))

    def set_model_info(self):
        super().set_model_info()
        self.Method = self.get_info_value('Method')


AlgorithmMapping = dict(textrecog=RecogAlgorithmInfo, textdet=DetAlgorithmInfo)


class BaseAlgorithmPaperList:
    overview_header = ['Title', 'Venue', 'Year', 'Institution', 'URL', 'Code']

    def __init__(self, paper_root, task, save_dir) -> None:
        papers_dir = osp.join(paper_root, task, 'algorithm')
        paper_paths = os.listdir(papers_dir)
        paper_paths.sort()
        self.algorithm_list = [
            AlgorithmMapping[task](yaml.safe_load(
                open(osp.join(papers_dir, paper_path))))
            for paper_path in paper_paths
        ]

        self.task = task
        self.papers_dir = save_dir

        self.algorithm_dir = osp.join(self.papers_dir, 'algorithm')
        os.makedirs(self.algorithm_dir, exist_ok=True)

        self.table_cfg = dict(
            tablefmt='pipe',
            floatfmt='.2f',
            numalign='right',
            stralign='center')

    def gen_algorithm(self):
        self.gen_algorithm_single(osp.join(self.algorithm_dir, 'paper'))
        self.gen_algorithm_overview(
            osp.join(self.algorithm_dir, 'overview.md'))
        # self.gen_algorithm_venue(osp.join(self.algorithm_dir, 'venue.md'))
        # self.gen_algorithm_method(osp.join(self.algorithm_dir, 'method.md'))
        self.gen_sota(osp.join(self.algorithm_dir, 'sota.md'))

    def gen_algorithm_overview(self, save_path):
        rows = list()
        for paper in self.algorithm_list:
            row = [getattr(paper, head) for head in self.overview_header]
            rows.append(row)

        with open(save_path, 'w') as f:
            f.write('# Overview\n')
            f.write('```{table}\n:class: model-summary nowrap field-list '
                    'table table-hover\n')
            f.write(tabulate(rows, self.overview_header, **self.table_cfg))
            f.write('\n```\n')

    def gen_sota(self, save_path):
        pass

    def gen_algorithm_single(self, paper_dirs):
        pass


class TextRecogPaperList(BaseAlgorithmPaperList):
    sota_header = ['Abbr', 'Venue', 'Year', 'TrainDataset']

    def gen_sota(self, save_path):
        benchmark_dataset_list = [
            'IIIT5K', 'SVT', 'IC13', 'IC15', 'SVTP', 'CUTE'
        ]
        metric_name = 'WAICS'
        rows = list()
        for paper in self.algorithm_list:
            results = [
                getattr(paper, f'{metric_name}_{dataset}', 'N/A')
                for dataset in benchmark_dataset_list
            ]

            # remove 'N/A' in avg_list
            avg_list = [x for x in results if x != 'N/A']
            # if all results are 'N/A', we don't append it to the sota table
            if len(avg_list) != 0:
                avg = sum(avg_list) / len(avg_list)
            else:
                continue
            results.append(avg)
            row = [getattr(paper, head) for head in self.sota_header] + results
            rows.append(row)
        # sort average accuracy from small to large
        rows = sorted(rows, key=lambda x: x[-1])
        with open(save_path, 'w') as f:
            f.write('# SOTA\n')
            f.write(
                '```{table}\n:class: model-summary nowrap field-list table '
                'table-hover\n')
            header = self.sota_header + benchmark_dataset_list + ['Avg']
            f.write(tabulate(rows, header, **self.table_cfg))
            f.write('\n```\n')

    def gen_algorithm_single(self, paper_dirs):
        overview_header = ['Venue', 'Year', 'Institution', 'URL', 'Code']
        model_header = [
            'Architecture', 'Learning Method', 'Language Modality', 'Times',
            'FLOPs', 'Params'
        ]
        if not os.path.exists(paper_dirs):
            os.makedirs(paper_dirs)
        for paper in self.algorithm_list:
            file_name = paper.BaseTitle + '.md'
            file_path = os.path.join(paper_dirs, file_name)

            with open(file_path, 'w') as f:
                f.write('# {}\n'.format(paper.BaseTitle))

                f.write('## Overview\n\n\n')
                row = [getattr(paper, head, 'N/A') for head in overview_header]

                f.write(tabulate([row], overview_header, **self.table_cfg))
                f.write('\n\n')
                f.write('## Model\n\n')
                f.write('### Abstract\n\n')
                f.write(paper['Abstract'] + '\n\n')
                f.write(paper.Network)
                f.write('### Model information\n\n')

                rows = [
                    getattr(paper, head.replace(' ', '_'), 'N/A')
                    for head in model_header
                ]
                f.write(tabulate([rows], model_header, **self.table_cfg))
                f.write('\n\n')
                f.write('## Results\n\n')

                results_header = ['Metric', 'Training DataSets'
                                  ] + paper.test_dataset_list

                rows = list()
                for metric in paper.metric_list:

                    row = [
                        metric,
                        paper.TrainDataset,
                    ]
                    row += [
                        getattr(paper, f'{metric}_{key}')
                        for key in paper.test_dataset_list
                    ]
                    rows.append(row)
                f.write(tabulate(rows, results_header, **self.table_cfg))
                f.write('\n\n')
                f.write('## Citation\n\n')
                f.write('```bibtex\n{}\n```\n'.format(paper['Bibtex']))


class TextDetPaperList(BaseAlgorithmPaperList):
    sota_header = ['Abbr', 'Venue', 'Year']

    def gen_sota(self, save_path):
        benchmark_dataset_list = ['ICDAR2015', 'CTW500']
        metric_list = ['Precision', 'Recall', 'Hmean']
        rows = list()
        for paper in self.algorithm_list:
            results = [
                getattr(paper, f'{metric_name}_{dataset}', 'N/A')
                for dataset in benchmark_dataset_list
                for metric_name in metric_list
            ]

            row = [getattr(paper, head) for head in self.sota_header] + results
            rows.append(row)
        # sort average accuracy from small to large
        rows = sorted(rows, key=lambda x: x[-1])
        with open(save_path, 'w') as f:
            f.write('# SOTA\n')
            f.write(
                '```{table}\n:class: model-summary nowrap field-list table '
                'table-hover\n')
            header = self.sota_header + [
                f'{dataset}_{metric_name}'
                for dataset in benchmark_dataset_list
                for metric_name in metric_list
            ]
            f.write(tabulate(rows, header, **self.table_cfg))
            f.write('\n```\n')

    def gen_algorithm_single(self, paper_dirs):
        overview_header = ['Venue', 'Year', 'Institution', 'URL', 'Code']
        model_header = ['Method', 'FPS', 'FLOPs', 'Params']
        if not os.path.exists(paper_dirs):
            os.makedirs(paper_dirs)
        for paper in self.algorithm_list:
            file_name = paper.BaseTitle + '.md'
            file_path = os.path.join(paper_dirs, file_name)

            with open(file_path, 'w') as f:
                f.write('# {}\n'.format(paper.BaseTitle))

                f.write('## Overview\n\n\n')
                row = [getattr(paper, head, 'N/A') for head in overview_header]

                f.write(tabulate([row], overview_header, **self.table_cfg))
                f.write('\n\n')
                f.write('## Model\n\n')
                f.write('### Abstract\n\n')
                f.write(paper['Abstract'] + '\n\n')
                f.write(paper.Network)
                f.write('### Model information\n\n')

                rows = [
                    getattr(paper, head.replace(' ', '_'), 'N/A')
                    for head in model_header
                ]
                f.write(tabulate([rows], model_header, **self.table_cfg))
                f.write('\n\n')
                f.write('## Results\n\n')

                results_header = ['Metric'] + paper.test_dataset_list

                rows = list()
                for metric in paper.metric_list:

                    row = [metric]
                    row += [
                        getattr(paper, f'{metric}_{key}')
                        for key in paper.test_dataset_list
                    ]
                    rows.append(row)
                f.write(tabulate(rows, results_header, **self.table_cfg))
                f.write('\n\n')
                f.write('## Citation\n\n')
                f.write('```bibtex\n{}\n```\n'.format(paper['Bibtex']))


papers_dir = '../../paper_zoo'
# papers_dir = 'paper_zoo'

save_dir = osp.join('paper_zoo', 'textrecog')
paper_list = TextRecogPaperList(papers_dir, 'textrecog', save_dir)
os.makedirs(save_dir, exist_ok=True)
paper_list.gen_algorithm()

save_dir = osp.join('paper_zoo', 'textdet')
paper_list = TextDetPaperList(papers_dir, 'textdet', save_dir)
os.makedirs(save_dir, exist_ok=True)
paper_list.gen_algorithm()
