_base_ = [
    '_base_marec_vit_s.py',
    '../_base_/datasets/union14m_train.py',
    '../_base_/datasets/union14m_benchmark.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamw_cos_6e.py',
]

model = dict(
    backbone=dict(
        type='VisionTransformer',
        img_size=(32, 128),
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        pretrained=None),
    decoder=dict(
        type='MAERecDecoder',
        n_layers=6,
        d_embedding=768,
        n_head=8,
        d_model=768,
        d_inner=3072,
        d_k=96,
        d_v=96))

# dataset settings
train_list = [
    _base_.union14m_challenging, _base_.union14m_hard, _base_.union14m_medium,
    _base_.union14m_normal, _base_.union14m_easy
]

test_list = [
    _base_.union14m_benchmark_artistic,
    _base_.union14m_benchmark_multi_oriented,
    _base_.union14m_benchmark_contextless,
    _base_.union14m_benchmark_curve,
    _base_.union14m_benchmark_incomplete,
    _base_.union14m_benchmark_incomplete_ori,
    _base_.union14m_benchmark_multi_words,
    _base_.union14m_benchmark_salient,
    _base_.union14m_benchmark_general,
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))

auto_scale_lr = dict(base_batch_size=64)

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)


train_dataloader = dict(
    batch_size=64,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader


val_evaluator = dict(dataset_prefixes=[
    'artistic', 'multi-oriented', 'contextless', 'curve', 'incomplete',
    'incomplete-ori', 'multi-words', 'salient', 'general'
])

test_evaluator = val_evaluator