from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmocr.registry import DATASETS

if __name__ == '__main__':
    cfg_path = '/Users/wangnu/Documents/GitHub/mmocr/projects/' \
        'LayoutLMv3/configs/layoutlmv3_xfund_zh.py'
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get('default_scope', 'mmocr'))

    dataset_cfg = cfg.train_dataset
    dataset_cfg['tokenizer'] = \
        '/Users/wangnu/Documents/GitHub/mmocr/data/layoutlmv3-base-chinese'

    train_pipeline = [
        dict(type='LoadImageFromFile', color_type='color'),
        dict(type='Resize', scale=(224, 224))
    ]
    dataset_cfg['pipeline'] = train_pipeline
    ds = DATASETS.build(dataset_cfg)
    data = ds[0]
    print('hi')
