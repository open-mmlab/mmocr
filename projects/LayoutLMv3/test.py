from functools import partial

from mmengine.config import Config
from mmengine.dataset.utils import COLLATE_FUNCTIONS
from mmengine.registry import DATA_SAMPLERS, init_default_scope
from torch.utils.data import DataLoader

from mmocr.registry import DATASETS, MODELS

if __name__ == '__main__':
    cfg_path = '/Users/wangnu/Documents/GitHub/mmocr/projects/' \
        'LayoutLMv3/configs/ser/layoutlmv3_xfund_zh.py'
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get('default_scope', 'mmocr'))

    pretrained_model = '/Users/wangnu/Documents/GitHub'
    '/mmocr/data/layoutlmv3-base-chinese'

    dataset_cfg = cfg.train_dataset
    dataset_cfg['tokenizer'] = dict(
        pretrained_model_name_or_path=pretrained_model, use_fast=True)
    train_pipeline = [
        dict(type='LoadImageFromFile', color_type='color'),
        dict(
            type='ProcessImageForLayoutLMv3',
            image_processor=dict(
                pretrained_model_name_or_path=pretrained_model,
                size=(224, 224),
                apply_ocr=False)),
        dict(
            type='ProcessTokenForLayoutLMv3',
            padding='max_length',
            max_length=512),
        dict(
            type='PackSERInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor',
                       'id2biolabel'))
    ]
    dataset_cfg['pipeline'] = train_pipeline
    train_dataset = DATASETS.build(dataset_cfg)

    model_cfg = dict(
        type='HFLayoutLMv3ForTokenClassificationWrapper',
        classifier=dict(
            pretrained_model_name_or_path=pretrained_model, num_labels=7),
        data_preprocessor=None)

    collate_fn_cfg = dict(type='default_collate')
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)
    collate_fn = partial(collate_fn, **collate_fn_cfg)

    sampler_cfg = dict(
        type='DefaultSampler', dataset=train_dataset, shuffle=True)
    sampler = DATA_SAMPLERS.build(sampler_cfg)

    from mmengine.dataset.utils import worker_init_fn as default_worker_init_fn
    init_fn = partial(
        default_worker_init_fn,
        num_workers=2,
        rank=0,
        seed=301967075,
        disable_subprocess_warning=False)

    train_dataloader = DataLoader(
        batch_size=1,
        dataset=train_dataset,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        worker_init_fn=init_fn)

    model = MODELS.build(model_cfg)

    for idx, data_batch in enumerate(train_dataloader):
        print(idx)
        result = model.forward(**data_batch, mode='loss')
        break

    print('Done')
