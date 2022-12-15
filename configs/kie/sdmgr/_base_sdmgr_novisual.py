num_classes = 26

model = dict(
    type='SDMGR',
    kie_head=dict(
        type='SDMGRHead',
        visual_dim=16,
        num_classes=num_classes,
        module_loss=dict(type='SDMGRModuleLoss'),
        postprocessor=dict(type='SDMGRPostProcessor')),
    dictionary=dict(
        type='Dictionary',
        dict_file='data/kie/wildreceipt/dict.txt',
        with_padding=True,
        with_unknown=True,
        unknown_token=None),
)

train_pipeline = [
    dict(type='LoadKIEAnnotations'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='PackKIEInputs')
]
test_pipeline = [
    dict(type='LoadKIEAnnotations'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='PackKIEInputs'),
]

val_evaluator = dict(
    type='F1Metric',
    mode='macro',
    num_classes=num_classes,
    ignored_classes=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25])
test_evaluator = val_evaluator
