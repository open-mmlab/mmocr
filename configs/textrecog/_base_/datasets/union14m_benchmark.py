union14m_root = 'data/Union14M-L/'
union14m_benchmark_root = 'data/Union14M-L/Union14M-Benchmarks'

union14m_benchmark_artistic = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{union14m_benchmark_root}/artistic'),
    ann_file=f'{union14m_benchmark_root}/artistic/annotation.json',
    test_mode=True,
    pipeline=None)

union14m_benchmark_contextless = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{union14m_benchmark_root}/contextless'),
    ann_file=f'{union14m_benchmark_root}/contextless/annotation.json',
    test_mode=True,
    pipeline=None)

union14m_benchmark_curve = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{union14m_benchmark_root}/curve'),
    ann_file=f'{union14m_benchmark_root}/curve/annotation.json',
    test_mode=True,
    pipeline=None)

union14m_benchmark_incomplete = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{union14m_benchmark_root}/incomplete'),
    ann_file=f'{union14m_benchmark_root}/incomplete/annotation.json',
    test_mode=True,
    pipeline=None)

union14m_benchmark_incomplete_ori = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{union14m_benchmark_root}/incomplete_ori'),
    ann_file=f'{union14m_benchmark_root}/incomplete_ori/annotation.json',
    test_mode=True,
    pipeline=None)

union14m_benchmark_multi_oriented = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{union14m_benchmark_root}/multi_oriented'),
    ann_file=f'{union14m_benchmark_root}/multi_oriented/annotation.json',
    test_mode=True,
    pipeline=None)

union14m_benchmark_multi_words = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{union14m_benchmark_root}/multi_words'),
    ann_file=f'{union14m_benchmark_root}/multi_words/annotation.json',
    test_mode=True,
    pipeline=None)

union14m_benchmark_salient = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{union14m_benchmark_root}/salient'),
    ann_file=f'{union14m_benchmark_root}/salient/annotation.json',
    test_mode=True,
    pipeline=None)

union14m_benchmark_general = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=f'{union14m_root}/'),
    ann_file=f'{union14m_benchmark_root}/general/annotation.json',
    test_mode=True,
    pipeline=None)
