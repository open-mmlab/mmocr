#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import functools as func
import glob
import re
from os.path import basename, splitext

import numpy as np
import titlecase


def title2anchor(name):
    return re.sub(r'-+', '-', re.sub(r'[^a-zA-Z0-9]', '-',
                                     name.strip().lower())).strip('-')


# Count algorithms

files = sorted(glob.glob('*_models.md'))

stats = []

for f in files:
    with open(f, 'r') as content_file:
        content = content_file.read()

    # Remove the blackquote notation from the paper link under the title
    # for better layout in readthedocs
    expr = r'(^## \s*?.*?\s+?)>\s*?(\[.*?\]\(.*?\))'
    content = re.sub(expr, r'\1\2', content, flags=re.MULTILINE)
    with open(f, 'w') as content_file:
        content_file.write(content)

    # title
    title = content.split('\n')[0].replace('#', '')

    # count papers
    exclude_papertype = ['ABSTRACT', 'IMAGE']
    exclude_expr = ''.join(f'(?!{s})' for s in exclude_papertype)
    expr = rf'<!-- \[{exclude_expr}([A-Z]+?)\] -->'\
        r'\s*\n.*?\btitle\s*=\s*{(.*?)}'
    papers = set(
        (papertype, titlecase.titlecase(paper.lower().strip()))
        for (papertype, paper) in re.findall(expr, content, re.DOTALL))
    print(papers)
    # paper links
    revcontent = '\n'.join(list(reversed(content.splitlines())))
    paperlinks = {}
    for _, p in papers:
        q = p.replace('\\', '\\\\').replace('?', '\\?')
        paper_link = title2anchor(
            re.search(
                rf'\btitle\s*=\s*{{\s*{q}\s*}}.*?\n## (.*?)\s*[,;]?\s*\n',
                revcontent, re.DOTALL | re.IGNORECASE).group(1))
        paperlinks[p] = f'[{p}]({splitext(basename(f))[0]}.html#{paper_link})'
    paperlist = '\n'.join(
        sorted(f'    - [{t}] {paperlinks[x]}' for t, x in papers))
    # count configs
    configs = set(x.lower().strip()
                  for x in re.findall(r'https.*configs/.*\.py', content))

    # count ckpts
    ckpts = set(x.lower().strip()
                for x in re.findall(r'https://download.*\.pth', content)
                if 'mmocr' in x)

    statsmsg = f"""
## [{title}]({f})

* Number of checkpoints: {len(ckpts)}
* Number of configs: {len(configs)}
* Number of papers: {len(papers)}
{paperlist}

    """

    stats.append((papers, configs, ckpts, statsmsg))

allpapers = func.reduce(lambda a, b: a.union(b), [p for p, _, _, _ in stats])
allconfigs = func.reduce(lambda a, b: a.union(b), [c for _, c, _, _ in stats])
allckpts = func.reduce(lambda a, b: a.union(b), [c for _, _, c, _ in stats])
msglist = '\n'.join(x for _, _, _, x in stats)

papertypes, papercounts = np.unique([t for t, _ in allpapers],
                                    return_counts=True)
countstr = '\n'.join(
    [f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])

modelzoo = f"""
# Statistics

* Number of checkpoints: {len(allckpts)}
* Number of configs: {len(allconfigs)}
* Number of papers: {len(allpapers)}
{countstr}

{msglist}
"""

with open('modelzoo.md', 'w') as f:
    f.write(modelzoo)
