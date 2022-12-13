#!/usr/bin/env python
import os
import os.path as osp
from typing import Dict, List

import yaml

TYPES = [
    'CTC-Based', 'Attention-Based', 'Transformer-Based', 'Self-Supervised',
    'Semi-Supervised', 'Language-Based', 'Dataset', 'Survey'
]

VENUES = [
    'CVPR', 'ICCV', 'ECCV', 'AAAI', 'IJCAI', 'ICDAR', 'ACMMM', 'NIPS', 'ICML',
    'arXiv'
]


def get_recog_paper_type(yaml_dict: Dict) -> List[str]:
    """Given a dict of a paper yaml file, return the type of the paper.

    Args:
        yaml_dict (dict): A dict of a paper yaml file.

    Returns:
        List[str]: Types of the paper. It should be in TYPES.
    """
    # PaperType Param
    type_list = []
    if 'Survey' in yaml_dict['PaperType']:
        type_list.append('Survey')
    if 'Dataset' in yaml_dict['PaperType']:
        type_list.append('Dataset')

    # Architecture Param
    if 'CTC' in yaml_dict['MODELS']['Architecture']:
        type_list.append('CTC-Based')
    if 'Attention' in yaml_dict['MODELS']['Architecture']:
        type_list.append('Attention-Based')
    if 'Transformer' in yaml_dict['MODELS']['Architecture']:
        type_list.append('Transformer-Based')

    # Learning Method:
    if 'Self-Supervised' in yaml_dict['MODELS']['Learning Method']:
        type_list.append('Self-Supervised')
    if 'Semi-Supervised' in yaml_dict['MODELS']['Learning Method']:
        type_list.append('Semi-Supervised')

    # Language Modality:
    if 'Explicit Language Model' in yaml_dict['MODELS']['Language Modality']:
        type_list.append('Language-Based')

    return type_list


def parse_recog_paper_yaml(path: str, idx: int) -> dict:
    """Given a path to a text recognition-related paper yaml file, return a
    dict of the paper info.

    Args:
        path (str): Path to a paper yaml file.
        idx (int): Index of the paper. Used to generate sota table.

    Returns:
        dict: A dict of paper info, including the following keys:
            - title (str): Title of the paper.
            - abbreviation (str): Abbreviation of the paper.
            - venue (str): Venue of the paper. It should be in VENUES.
            - type (List[str]): Type of the paper. It should be in TYPES.
            - year (str): Year of the paper.
            - avg_acc (str): Average accuracy on benchmarks.
            - results (dict): Results on different datasets.
            - index (str): Markdown Index of the paper. Used to query the paper
                in sota table.
            - mkdown (str): Markdown format of the paper.
    """
    meta = yaml.safe_load(open(path))
    infos = dict(
        title=meta['Title'],
        abbreviation=meta['Abbreviation'],
        venue=meta['Venue'],
        url=meta['URL'],
        paper_reading_url=meta['Paper Reading URL'],
        type=get_recog_paper_type(meta),
        year=meta['Year'],
        avg_acc=meta['MODELS']['Results']['Common Benchmarks']['Avg.'],
        results=meta['MODELS']['Results'],
        index=idx)

    mkdown = '\n<details close>\n'
    # add title
    mkdown += f'<summary id={infos["index"]}><strong>[{infos["venue"]}\'{infos["year"]}: {infos["abbreviation"]}]</strong> {infos["title"]}</summary>\n\n'  # noqa: E501
    # add sub-title
    mkdown += f'> {infos["title"]}, *{infos["venue"]}*, {infos["year"]}.\n\n'  # noqa: E501
    mkdown += '<div align="center">\n'
    root_path = osp.dirname(path)
    # add network structure
    mkdown += f'<img src="{os.path.join(root_path,meta["MODELS"]["Network Structure"])}" height="width=400"/>\n'  # noqa: E501
    mkdown += '</div>\n\n'
    # add paper link
    mkdown += f'- **Paper Link**: [Link]({infos["url"]})\n\n'
    # add paper read link is exists
    if meta['Paper Reading URL'] != 'N/A':
        mkdown += f'- **Paper Reading Link**: [Link]({meta["Paper Reading URL"]})\n\n'  # noqa: E501

    # add code link is exists
    if meta['Code'] != 'N/A':
        mkdown += f'- **Code**: [Link]({meta["Code"]})\n\n'
    else:
        mkdown += '- **Code**: N/A\n\n'

    # add abstract
    mkdown += f'- **Abstract:** {meta["Abstract"]}\n\n'

    # add results
    mkdown += '- **Results:**\n'
    mkdown += '    | IIIT | SVT | IC13 | IC15 | SVTP | CUTE | Avg. | FPS | FLOPS | PARAMS |  \n'  # noqa: E501
    mkdown += '    | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |  \n'  # noqa: E501
    table = f'|{meta["MODELS"]["Results"]["Common Benchmarks"]["IIIT"]}|' + \
        f'{meta["MODELS"]["Results"]["Common Benchmarks"]["SVT"]}  |' + \
        f'{meta["MODELS"]["Results"]["Common Benchmarks"]["IC13"]} |' + \
        f'{meta["MODELS"]["Results"]["Common Benchmarks"]["IC15"]} |' + \
        f'{meta["MODELS"]["Results"]["Common Benchmarks"]["SVTP"]} |' + \
        f'{meta["MODELS"]["Results"]["Common Benchmarks"]["CUTE"]} |' + \
        f'{meta["MODELS"]["Results"]["Common Benchmarks"]["Avg."]} |' + \
        f'{meta["MODELS"]["FPS"]["ITEM"]} |' + \
        f'{meta["MODELS"]["FLOPS"]["ITEM"]} |' + \
        f'{meta["MODELS"]["PARAMS"]} |\n'
    mkdown += table
    # add bibtex
    mkdown += f'- **BibTeX:** {meta["Bibtex"]}\n\n'
    mkdown += '\n</details>\n\n'
    infos['mkdown'] = mkdown

    return infos


def build_sota_table(papers: List[Dict]):
    """Build SOTA Table.

    Args:
        papers (List[Dict]): A list of paper info dicts.
    """
    # sort papers by avg_acc, from small to large
    papers = sorted(papers, key=lambda x: x['avg_acc'])
    table = '| Model | IIIT | SVT | IC13 | IC15 | SVTP | CUTE | Avg. |\n'  # noqa: E501
    table += '| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |\n'  # noqa: E501
    for paper in papers:
        table += f'|[{paper["abbreviation"]}](#{paper["index"]}) |' + \
            f'    {paper["results"]["Common Benchmarks"]["IIIT"]} |' + \
            f'    {paper["results"]["Common Benchmarks"]["SVT"]}  |' + \
            f'    {paper["results"]["Common Benchmarks"]["IC13"]} |' + \
            f'    {paper["results"]["Common Benchmarks"]["IC15"]} |' + \
            f'    {paper["results"]["Common Benchmarks"]["SVTP"]} |' + \
            f'    {paper["results"]["Common Benchmarks"]["CUTE"]} |' + \
            f'    {paper["results"]["Common Benchmarks"]["Avg."]} |\n'

    return table


paper_zoo_path = '../../paper_zoo/textrecog'
papers = []

# build contents
contents = '# Contents\n'
contents += ' - [SOTA Table](#sota-table)\n'
contents += ' - [Types](#types)\n'
for type in TYPES:
    contents += '   - [{}](#{})\n'.format(type, type.lower())
contents += ' - [Venues](#venues)\n'
for venue in VENUES:
    contents += '   - [{}](#{})\n'.format(venue, venue.lower())
contents += '----\n'
# parse papers
for idx, paper in enumerate(os.listdir(paper_zoo_path)):
    for file in os.listdir(osp.join(paper_zoo_path, paper)):
        if file.endswith('.yaml'):
            paper_info = parse_recog_paper_yaml(
                osp.join(paper_zoo_path, paper, file), idx)
            papers.append(paper_info)
# sort papers by the combination of year and venue, from newest to oldest
papers = sorted(
    papers, key=lambda x: (str(x['year']) + x['venue']), reverse=True)
# build sota table
sota_tables = '## SOTA Table\n'
sota_tables += build_sota_table(papers)
sota_tables += '----\n'
# build types
types = '## Types\n'
for type in TYPES:
    types += f'#### {type}\n'
    for paper in papers:
        if type in paper['type']:
            types += paper['mkdown']
    types += '----\n'
# build venues
venues = '## Venues\n'
for venue in VENUES:
    venues += f'#### {venue}\n'
    for paper in papers:
        if venue == paper['venue']:
            venues += paper['mkdown']
    venues += '----\n'
outputs = contents + sota_tables + types + venues
# write to paper_zoo.md
with open('paperzoo.md', 'w') as f:
    f.write(outputs)
