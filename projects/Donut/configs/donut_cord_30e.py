_base_ = [
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_adam_fp16.py',
]

data_root = 'datasets/cord-v2/data'
task_name='cord-v2'

custom_imports = dict(
    imports=['projects.Donut.donut'], allow_failed_imports=False)

# dictionary = dict(
#     type='Dictionary',
#     dict_file='{{ fileDirname }}/../../../dicts/english_digits_symbols.txt',
#     with_padding=True,
#     with_unknown=True,
#     same_start_end=True,
#     with_start=True,
#     with_end=True)

model = dict(
    type='Donut',
    data_preprocessor=dict(
        type='DonutDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    encoder=dict(
        type='SwinEncoder',
        input_size=[1280, 960],
        align_long_axis=False,
        window_size=10,
        encoder_layer=[2, 2, 14, 2],
        init_cfg=dict(type='Pretrained', checkpoint='data/donut_base_encoder.pth')
        ),
    decoder=dict(
        type='BARTDecoder',
        max_position_embeddings=None,
        task_start_token=f"<s_{task_name}>",
        prompt_end_token=f"<s_{task_name}>",
        decoder_layer=4,
        tokenizer_cfg=dict(type='XLMRobertaTokenizer', checkpoint='naver-clova-ix/donut-base'),
        init_cfg=dict(type='Pretrained', checkpoint='data/donut_base_decoder.pth')),
    sort_json_key = False,
    )

train_pipeline = [
    dict(type='LoadImageFromFile', ignore_empty=True, min_size=2),
    dict(type='LoadJsonAnnotations', with_bbox=False, with_label=False),
    dict(type='TorchVisionWrapper', op='Resize', size=960, max_size=1280),
    dict(type='RandomPad', input_size=[1280, 960], random_padding=True),
    dict(
        type='PackKIEInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'parses_json'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TorchVisionWrapper', op='Resize', size=960, max_size=1280),
    dict(type='RandomPad', input_size=[1280, 960], random_padding=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadJsonAnnotations', with_bbox=False, with_label=False),
    dict(
        type='PackKIEInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'parses_json'))
]

# dataset settings
train_dataset = dict(
    type='CORDDataset',
    data_root=data_root,
    split_name='train',
    pipeline=train_pipeline)
val_dataset = dict(
    type='CORDDataset',
    data_root=data_root,
    split_name='validation',
    pipeline=test_pipeline)
test_dataset = dict(
    type='CORDDataset',
    data_root=data_root,
    split_name='test',
    pipeline=test_pipeline)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader

val_evaluator = dict(
    type='DonutValEvaluator',
    key='parses')
test_evaluator = dict(
    type='JSONParseEvaluator',
    key='parses_json')

randomness = dict(seed=2022)
find_unused_parameters = True
