#!/usr/bin/env python
import os.path as osp
import re

# This script reads /projects/selected.txt and generate projectzoo.md

files = []

project_zoo = """
# Models in Projects

Here are some selected project implementations that are not yet included in
MMOCR package, but are ready to use.

"""

files = open('../../projects/selected.txt').readlines()

for file in files:
    file = file.strip()
    with open(osp.join('../../', file)) as f:
        content = f.read()

    # Extract title
    expr = '# (.*?)\n'
    title = re.search(expr, content).group(1)
    project_zoo += f'## {title}\n\n'

    # Locate the description
    expr = '## Description\n(.*?)##'
    description = re.search(expr, content, re.DOTALL).group(1)
    project_zoo += f'{description}\n'

    # check milestone 1
    expr = r'- \[(.?)\] Milestone 1'
    state = re.search(expr, content, re.DOTALL).group(1)
    infer_state = '✔' if state == 'x' else '❌'

    # check milestone 2
    expr = r'- \[(.?)\] Milestone 2'
    state = re.search(expr, content, re.DOTALL).group(1)
    training_state = '✔' if state == 'x' else '❌'

    # add table
    readme_link = f'https://github.com/open-mmlab/mmocr/blob/dev-1.x/{file}'
    project_zoo += '### Status \n'
    project_zoo += '| Inference | Train | README |\n'
    project_zoo += '| --------- | -------- | ------ |\n'
    project_zoo += f'|️{infer_state}|{training_state}|[link]({readme_link})|\n'

with open('projectzoo.md', 'w') as f:
    f.write(project_zoo)
