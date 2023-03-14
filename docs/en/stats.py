#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import functools as func
import re
from os.path import basename, splitext

import numpy as np
import titlecase
from weight_list import gen_weight_list


def title2anchor(name):
    return re.sub(r'-+', '-', re.sub(r'[^a-zA-Z0-9]', '-',
                                     name.strip().lower())).strip('-')


# Count algorithms

files = [
    'backbones.md', 'textdet_models.md', 'textrecog_models.md', 'kie_models.md'
]

stats = []

for f in files:
    with open(f) as content_file:
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
    papers = {(papertype, titlecase.titlecase(paper.lower().strip()))
              for (papertype, paper) in re.findall(expr, content, re.DOTALL)}
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
        paperlinks[p] = f'[{p}]({splitext(basename(f))[0]}.md#{paper_link})'
    paperlist = '\n'.join(
        sorted(f'    - [{t}] {paperlinks[x]}' for t, x in papers))
    # count configs
    configs = {
        x.lower().strip()
        for x in re.findall(r'https.*configs/.*\.py', content)
    }

    # count ckpts
    ckpts = {
        x.lower().strip()
        for x in re.findall(r'https://download.*\.pth', content)
        if 'mmocr' in x
    }

    statsmsg = f"""
### [{title}]({f})

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

# get model list
weight_list = gen_weight_list()

modelzoo = f"""
# Overview

## Weights

Here are the list of weights available for
[Inference](user_guides/inference.md).

For the ease of reference, some weights may have shorter aliases, which will be
separated by `/` in the table.
For example, "`DB_r18 / dbnet_resnet18_fpnc_1200e_icdar2015`" means that you can
use either `DB_r18` or `dbnet_resnet18_fpnc_1200e_icdar2015`
to initialize the Inferencer:

```python
>>> from mmocr.apis import TextDetInferencer
>>> inferencer = TextDetInferencer(model='DB_r18')
>>> # equivalent to
>>> inferencer = TextDetInferencer(model='dbnet_resnet18_fpnc_1200e_icdar2015')
```

{weight_list}

## Statistics

* Number of checkpoints: {len(allckpts)}
* Number of configs: {len(allconfigs)}
* Number of papers: {len(allpapers)}
{countstr}

{msglist}
"""  # noqa

with open('modelzoo.md', 'w') as f:
    f.write(modelzoo)
