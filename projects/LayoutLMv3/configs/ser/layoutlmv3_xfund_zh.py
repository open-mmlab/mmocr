_base_ = [
    '/Users/wangnu/Documents/GitHub/mmocr/'
    'configs/ser/_base_/datasets/xfund_zh.py', '../_base_/default_runtime.py'
]

# specify a pretrained model
pretrained_model = '/Users/wangnu/Documents'
'/GitHub/mmocr/data/layoutlmv3-base-chinese'
# set classes
classes = ('answer', 'header', 'question', 'other')

# optimizer
max_epochs = 10
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=1e-3))
param_scheduler = [
    dict(type='PolyLR', power=0.9, end=max_epochs),
]
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

train_dataset = _base_.xfund_zh_ser_train
# specify a tokenizer for the dataset
train_dataset['tokenizer'] = dict(
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
        type='ProcessTokenForLayoutLMv3', padding='max_length',
        max_length=512),
    dict(
        type='PackSERInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'id2biolabel'))
]
train_dataset.pipeline = train_pipeline
# set collate_fn='default_collate' for the dataloader
collate_fn = dict(type='default_collate')
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=collate_fn,
    dataset=train_dataset)

model = dict(
    type='HFLayoutLMv3ForTokenClassificationWrapper',
    classifier=dict(
        pretrained_model_name_or_path=pretrained_model,
        num_labels=len(classes) * 2 - 1),
    data_preprocessor=None)

val_evaluator = None
test_evaluator = None
