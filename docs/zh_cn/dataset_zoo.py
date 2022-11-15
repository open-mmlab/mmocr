#!/usr/bin/env python
import os
import os.path as osp

import yaml

dataset_zoo_path = '../../dataset_zoo'
datasets = os.listdir(dataset_zoo_path)
datasets.sort()

table = '# 支持数据集一览\n'
table += '## 支持的数据集\n'
table += '| 数据集名称 | 文本检测 | 文本识别 | 端到端文本检测识别 | 关键信息抽取 |\n' \
         '|----------|---------|--------|------------------|-----------|\n'
details = '## 数据集详情\n'

for dataset in datasets:
    meta = yaml.safe_load(
        open(osp.join(dataset_zoo_path, dataset, 'metafile.yml')))
    dataset_name = meta['Name']
    paper = meta['Paper']
    data = meta['Data']

    table += '| [{}](#{}) | {} | {} | {} | {} |\n'.format(
        dataset,
        dataset_name.lower().replace(' ', '-'),
        '✓' if 'textdet' in data['Tasks'] else '',
        '✓' if 'textrecog' in data['Tasks'] else '',
        '✓' if 'textspotting' in data['Tasks'] else '',
        '✓' if 'kie' in data['Tasks'] else '',
    )

    details += '### {}\n'.format(dataset_name)
    details += "> \"{}\", *{}*, {}. [PDF]({})\n\n".format(
        paper['Title'], paper['Venue'], paper['Year'], paper['URL'])
    # Basic Info
    details += 'A. 数据集基础信息\n'
    details += ' - 官方网址: [{}]({})\n'.format(dataset, data['Website'])
    details += ' - 发布年份: {}\n'.format(paper['Year'])
    details += ' - 语言: {}\n'.format(data['Language'])
    details += ' - 场景: {}\n'.format(data['Scene'])
    details += ' - 标注粒度: {}\n'.format(data['Granularity'])
    details += ' - 支持任务: {}\n'.format(data['Tasks'])
    details += ' - 数据集许可证: [{}]({})\n\n'.format(data['License']['Type'],
                                                data['License']['Link'])

    # Format
    details += '<details> <summary>B. 标注格式</summary>\n\n</br>'
    sample_path = osp.join(dataset_zoo_path, dataset, 'sample_anno.md')
    if osp.exists(sample_path):
        with open(sample_path, 'r') as f:
            samples = f.readlines()
            samples = ''.join(samples)
            details += samples
    details += '</details>\n\n</br>'

    # Reference
    details += 'C. 参考文献\n'
    details += '```bibtex\n{}\n```\n'.format(paper['BibTeX'])

datasetzoo = table + details

with open('user_guides/data_prepare/datasetzoo.md', 'w') as f:
    f.write(datasetzoo)
