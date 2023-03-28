from mmengine.config import Config

from mmocr.registry import DATASETS

if __name__ == '__main__':
    cfg_path = '/Users/wangnu/Documents/GitHub/mmocr/projects/' \
        'LayoutLMv3/configs/layoutlmv3_xfund_zh.py'
    cfg = Config.fromfile(cfg_path)

    dataset_cfg = cfg.train_dataset
    dataset_cfg['tokenizer'] = \
        '/Users/wangnu/Documents/GitHub/mmocr/data/layoutlmv3-base-chinese'
    ds = DATASETS.build(dataset_cfg)
    print(ds[0])
