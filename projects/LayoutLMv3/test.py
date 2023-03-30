from functools import partial

from mmengine.config import Config
from mmengine.dataset.utils import COLLATE_FUNCTIONS
from mmengine.registry import init_default_scope
from torch.utils.data import DataLoader

from mmocr.registry import DATASETS, MODELS

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
        dict(type='Resize', scale=(224, 224),
             backend='pillow'),  # backend=pillow 数值与huggingface对齐
        dict(
            type='PackSERInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ]
    dataset_cfg['pipeline'] = train_pipeline
    train_dataloader_cfg = dict(
        batch_size=1,
        num_workers=8,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dataset_cfg)

    model_cfg = dict(
        type='LayoutLMv3TokenClassifier',
        backbone=dict(),
        cls_head=dict(),
        data_preprocessor=dict(
            type='LayoutLMv3DataPreprocessor',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            bgr_to_rgb=True))

    train_dataset = DATASETS.build(dataset_cfg)
    collate_fn_cfg = dict(type='pseudo_collate')
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)
    collate_fn = partial(collate_fn, **collate_fn_cfg)
    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=collate_fn)

    model = MODELS.build(model_cfg)

    for idx, data_batch in enumerate(train_dataloader):
        result = model.forward(data_batch)
        break

    print('Done')
