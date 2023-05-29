_base_ = [
    '../_base_/datasets/xfund_zh.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamw_1k.py'
]

# ================== Frequently modified parameters ==================
hf_pretrained_model = '/Users/wangnu/Documents/GitHub' \
    '/mmocr/data/layoutlmv3-base-chinese'
dataset_name = 'xfund_zh'
class_name = ('answer', 'header', 'question', 'other')
max_iters = 1000
val_interval = 100
lr = 7e-5
train_batch_size_per_gpu = 2
train_num_workers = 8
test_batch_size_per_gpu = 1  # can't batch infer now
test_num_workers = 8
only_label_first_subword = True  # select label process strategy
# ====================================================================
# =========================== schedule ===============================
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))
param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=lr,
        by_epoch=False,
        total_steps=max_iters,
        three_phase=True,
        final_div_factor=4),
]
# ====================================================================
# =========================== Dataset ================================
train_dataset = _base_.xfund_zh_ser_train
test_dataset = _base_.xfund_zh_ser_test
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(
        type='LoadProcessorFromPretrainedModel',
        pretrained_model_name_or_path=hf_pretrained_model,
        image_processor=dict(size=(224, 224), apply_ocr=False)),
    dict(type='ProcessImageForLayoutLMv3'),
    dict(
        type='ProcessTokenForLayoutLMv3',
        padding='max_length',
        max_length=512,
        truncation=True),
    dict(
        type='ConvertBIOLabelForSER',
        classes=class_name,
        only_label_first_subword=only_label_first_subword),
    dict(
        type='PackSERInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(
        type='LoadProcessorFromPretrainedModel',
        pretrained_model_name_or_path=hf_pretrained_model,
        image_processor=dict(size=(224, 224), apply_ocr=False)),
    dict(type='ProcessImageForLayoutLMv3'),
    dict(
        type='ProcessTokenForLayoutLMv3',
        padding='max_length',
        max_length=512,
        truncation=True),
    dict(
        type='ConvertBIOLabelForSER',
        classes=class_name,
        only_label_first_subword=only_label_first_subword),
    dict(
        type='PackSERInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'truncation_word_ids', 'instances'))
]
train_dataset.pipeline = train_pipeline
test_dataset.pipeline = test_pipeline
# ====================================================================
# ========================= Dataloader ===============================
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    collate_fn=dict(type='ser_collate', training=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=test_num_workers,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='ser_collate', training=False),
    dataset=test_dataset)
test_dataloader = val_dataloader
# ====================================================================
# ============================ Model =================================
model = dict(
    type='HFLayoutLMv3ForTokenClassificationWrapper',
    layoutlmv3_token_classifier=dict(
        pretrained_model_name_or_path=hf_pretrained_model,
        num_labels=len(class_name) * 2 - 1),
    loss_processor=dict(type='ComputeLossAfterLabelSmooth'),
    postprocessor=dict(
        type='SERPostprocessor',
        classes=class_name,
        only_label_first_subword=only_label_first_subword))
# ====================================================================
# ========================= Evaluation ===============================
val_evaluator = dict(type='SeqevalMetric', prefix=dataset_name)
test_evaluator = val_evaluator
# ====================================================================
# ======================= Visualization ==============================
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SERLocalVisualizer', name='visualizer', vis_backends=vis_backends)
# ====================================================================
# ============================= Hook =================================
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(
        type='CheckpointHook',
        interval=500,
        save_best=f'{dataset_name}/f1',
        rule='greater'),
    visualization=dict(
        type='VisualizationHook',
        interval=10,
        enable=True,
        show=False,
        draw_gt=True,
        draw_pred=True),
)
# ====================================================================
# ========================= Custom imports ===========================
custom_imports = dict(
    imports=[
        'projects.LayoutLMv3.datasets', 'projects.LayoutLMv3.evaluation',
        'projects.LayoutLMv3.models', 'projects.LayoutLMv3.visualization'
    ],
    allow_failed_imports=False)
# ====================================================================
