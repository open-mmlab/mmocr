#!/usr/bin/env python
import os.path as osp

from mmengine.fileio import load
from mmengine.utils import get_installed_path

content = '\n## Model List'

abbr2task = {
    'textdet': 'Text Detection',
    'textrecog': 'Text Recognition',
    'kie': 'Key Information Extraction',
}


def get_task(task_name: str):
    package_path = get_installed_path('mmocr')
    meta_indexes = load(osp.join(package_path, '.mim', 'model-index.yml'))
    for meta_path in meta_indexes['Import']:
        meta_path = osp.join(package_path, '.mim', meta_path)
        metainfo = load(meta_path)
        collection2md = {}
        for item in metainfo['Collections']:
            # adding a slash ahead of the README file path
            collection2md[item['Name']] = f'/{item["README"]}'
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
            yield name, readme


for abbr, task_name in abbr2task.items():
    content += f'\n### {task_name}\n'
    content += '| Model Name | Readme |\n' \
               '|------------|------|\n'  # noqa: E501
    for name, readme in get_task(abbr):
        content += f'| {name} | [link]({readme}) |\n'

with open('user_guides/inference.md', 'a') as f:
    f.write(content)
