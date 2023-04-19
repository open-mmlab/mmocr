from functools import partial

from mmengine.config import Config
from mmengine.dataset.utils import COLLATE_FUNCTIONS
from mmengine.registry import DATA_SAMPLERS, init_default_scope
from torch.utils.data import DataLoader

from mmocr.registry import DATASETS, MODELS

if __name__ == '__main__':
    cfg_path = '/Users/wangnu/Documents/GitHub/mmocr' \
        '/configs/ser/_base_/datasets/xfund_zh.py'
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get('default_scope', 'mmocr'))

    pretrained_model = '/Users/wangnu/Documents/GitHub/' \
        'mmocr/data/layoutlmv3-base-chinese'
    classes = ('answer', 'header', 'question', 'other')

    dataset_cfg = cfg.xfund_zh_ser_train
    train_pipeline = [
        dict(type='LoadImageFromFile', color_type='color'),
        dict(
            type='LoadProcessorFromPretrainedModel',
            pretrained_model_name_or_path=pretrained_model,
            image_processor=dict(size=(224, 224), apply_ocr=False),
            tokenizer=dict()),
        dict(type='ProcessImageForLayoutLMv3'),
        dict(
            type='ProcessTokenForLayoutLMv3',
            padding='max_length',
            max_length=512,
            truncation=True),
        dict(
            type='ConvertBIOLabelForSER',
            classes=classes,
            only_label_first_subword=True),
        dict(
            type='PackSERInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ]
    dataset_cfg['pipeline'] = train_pipeline
    train_dataset = DATASETS.build(dataset_cfg)

    model_cfg = dict(
        type='HFLayoutLMv3ForTokenClassificationWrapper',
        layoutlmv3_token_classifier=dict(
            pretrained_model_name_or_path=pretrained_model, num_labels=7),
        postprocessor=dict(type='SERPostprocessor'))

    collate_fn_cfg = dict(type='default_collate')
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)
    collate_fn = partial(collate_fn, **collate_fn_cfg)

    sampler_cfg = dict(
        type='DefaultSampler', dataset=train_dataset, shuffle=True)
    sampler = DATA_SAMPLERS.build(sampler_cfg)

    train_dataloader = DataLoader(
        batch_size=2,
        dataset=train_dataset,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=8)

    model = MODELS.build(model_cfg)

    for idx, data_batch in enumerate(train_dataloader):
        # result = model.forward(**data_batch, mode='loss')
        result = model.forward(**data_batch, mode='predict')
        break

    print('Done')
