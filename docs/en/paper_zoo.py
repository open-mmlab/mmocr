#!/usr/bin/env python
import os
import os.path as osp

import yaml
from tabulate import tabulate

VENUES = [
    'CVPR', 'ICCV', 'ECCV', 'AAAI', 'IJCAI', 'ICDAR', 'ACMMM', 'NIPS', 'ICML',
    'arXiv'
]
PAPER_TYPE = ['Algorithm', 'Dataset', 'Survey']


class BasePaperList:

    def __init__(self, data_root, task) -> None:
        papers_dir = osp.join(data_root, task)
        paper_paths = os.listdir(papers_dir)
        paper_paths.sort()
        paper_list = [
            yaml.safe_load(open(osp.join(papers_dir, paper_path)))
            for paper_path in paper_paths
        ]
        self.paper_list = paper_list
        self.algorithm_list = [
            p for p in self.paper_list if 'Algorithm' in p['PaperType']
        ]
        self.venues = VENUES

    def gen_model_overview(self, save_path):

        rows = list()
        for paper in self.algorithm_list:
            title = paper['Title']
            venue = paper['Venue']
            year = paper['Year']
            url = '[link]({})'.format(paper['URL'])
            code = '[link](' + paper['Code'] + ')' if paper[
                'Code'] != 'N/A' else 'N/A'
            rows.append([title, venue, year, url, code])

        with open(save_path, 'w') as f:
            f.write('# Overview\n')
            f.write("""```{table}\n:class: model-summary\n""")
            header = ['Title', 'Venue', 'Year', 'Paper', 'Code']
            table_cfg = dict(
                tablefmt='pipe',
                floatfmt='.2f',
                numalign='right',
                stralign='center')
            f.write(tabulate(rows, header, **table_cfg))
            f.write('\n```\n')

    def gen_venue_overview(self, save_path):
        venues_list = [list() for _ in range(len(self.venues))]
        other_list = list()
        for paper in self.algorithm_list:
            if paper['Venue'] in self.venues:
                venues_list[self.venues.index(paper['Venue'])].append(paper)
            else:
                other_list.append(paper)
        with open(save_path, 'w') as f:
            f.write('# Venue\n')
            for venue, venue_list in zip(self.venues, venues_list):
                if len(venue_list) == 0:
                    continue
                f.write('## {}\n'.format(venue))
                rows = list()
                for paper in venue_list:
                    title = paper['Title']
                    year = paper['Year']
                    url = '[link]({})'.format(paper['URL'])
                    code = '[link](' + paper['Code'] + ')' if paper[
                        'Code'] != 'N/A' else 'N/A'
                    rows.append([title, year, url, code])
                f.write("""```{table}\n:class: model-summary\n""")
                header = ['Title', 'Year', 'Paper', 'Code']
                table_cfg = dict(
                    tablefmt='pipe',
                    floatfmt='.2f',
                    numalign='right',
                    stralign='center')
                f.write(tabulate(rows, header, **table_cfg))
                f.write('\n```\n')
            if len(other_list) > 0:
                f.write('## Others\n')
                rows = list()
                for paper in other_list:
                    title = paper['Title']
                    venue = paper['Venue']
                    year = paper['Year']
                    url = '[link]({})'.format(paper['URL'])
                    code = '[link](' + paper['Code'] + ')' if paper[
                        'Code'] != 'N/A' else 'N/A'
                    rows.append([title, venue, year, url, code])
                f.write("""```{table}\n:class: model-summary\n""")
                header = ['Title', 'Venue', 'Year', 'Paper', 'Code']
                table_cfg = dict(
                    tablefmt='pipe',
                    floatfmt='.2f',
                    numalign='right',
                    stralign='center')
                f.write(tabulate(rows, header, **table_cfg))
                f.write('\n```\n')

    def gen_sota(self, save_path):
        pass


class TextRecogPaperList(BasePaperList):

    def gen_sota(self, save_path):
        """Build SOTA Table.

        Args:
            papers (List[Dict]): A list of paper info dicts.
        """
        # sort papers by avg_acc, from small to large
        benchmark_dataset_list = [
            'IIIT5K', 'SVT', 'IC13', 'IC15', 'SVTP', 'CUTE'
        ]
        metric_name = 'WAICS'
        rows = list()
        for paper in self.algorithm_list:
            title = paper['Abbreviation']
            venue = paper['Venue']
            test_dataset_res = paper['MODELS']['Experiment']['Test DataSets']
            results = list()
            avg_list = list()

            for dataset in benchmark_dataset_list:
                if dataset in test_dataset_res and metric_name in test_dataset_res[  # noqa: E501
                        dataset]:
                    results.append(test_dataset_res[dataset][metric_name])
                    avg_list.append(test_dataset_res[dataset][metric_name])
                else:
                    results.append('N/A')
            avg = sum(avg_list) / len(avg_list)
            results.append(avg)
            rows.append([title, venue] + results)

        with open(save_path, 'w') as f:
            f.write('# SOTA\n')
            f.write("""```{table}\n:class: model-summary\n""")
            header = ['Title', 'Venue'] + benchmark_dataset_list + ['Avg']
            table_cfg = dict(
                tablefmt='pipe',
                floatfmt='.2f',
                numalign='right',
                stralign='center')
            f.write(tabulate(rows, header, **table_cfg))
            f.write('\n```\n')


papers_dir = '../../paper_zoo'
paper_list = TextRecogPaperList(papers_dir, 'textrecog')
os.makedirs('paper_zoo/textrecog', exist_ok=True)
paper_list.gen_model_overview('paper_zoo/textrecog/overview.md')
paper_list.gen_sota('paper_zoo/textrecog/sota.md')
paper_list.gen_venue_overview('paper_zoo/textrecog/venues.md')
