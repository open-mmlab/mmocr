_base_ = ['../../_base_/default_runtime.py']

optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adam', weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning rate
param_scheduler = [
    dict(type='MultiStepLR', milestones=[40, 50], end=60),
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=100), )

num_classes = 4
key_node_idx = 1
value_node_idx = 2

model = dict(
    type='SDMGR',
    kie_head=dict(
        type='SDMGRHead',
        visual_dim=16,
        num_classes=num_classes,
        module_loss=dict(type='SDMGRModuleLoss'),
        postprocessor=dict(
            type='SDMGRPostProcessor',
            link_type='one-to-many',
            key_node_idx=key_node_idx,
            value_node_idx=value_node_idx)),
    dictionary=dict(
        type='Dictionary',
        dict_file='data/wildreceipt/dict.txt',
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
    dict(
        type='LoadKIEAnnotations',
        key_node_idx=key_node_idx,
        value_node_idx=value_node_idx),  # Keep key->value edges for evaluation
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='PackKIEInputs'),
]

dataset_type = 'WildReceiptDataset'
data_root = 'data/wildreceipt/'

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=data_root + 'class_list.txt',
    ann_file='openset_train.txt',
    pipeline=train_pipeline)

test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=data_root + 'class_list.txt',
    ann_file='openset_test.txt',
    test_mode=True,
    pipeline=test_pipeline)

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='F1Metric',
        prefix='node',
        key='labels',
        mode=['micro', 'macro'],
        num_classes=num_classes,
        cared_classes=[key_node_idx, value_node_idx]),
    dict(
        type='F1Metric',
        prefix='edge',
        mode='micro',
        key='edge_labels',
        cared_classes=[1],  # binary f1 score
        num_classes=2)
]
test_evaluator = val_evaluator
