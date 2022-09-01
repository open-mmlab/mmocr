_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/wildreceipt-openset.py',
    '../_base_/schedules/schedule_adam_60e.py',
    '_base_sdmgr_novisual.py',
]

node_num_classes = 4  # 4 classes: bg, key, value and other
edge_num_classes = 2  # edge connectivity
key_node_idx = 1
value_node_idx = 2

model = dict(
    type='SDMGR',
    kie_head=dict(
        num_classes=node_num_classes,
        postprocessor=dict(
            link_type='one-to-many',
            key_node_idx=key_node_idx,
            value_node_idx=value_node_idx)),
)

test_pipeline = [
    dict(
        type='LoadKIEAnnotations',
        key_node_idx=key_node_idx,
        value_node_idx=value_node_idx),  # Keep key->value edges for evaluation
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='PackKIEInputs'),
]

wildreceipt_openset_train = _base_.wildreceipt_openset_train
wildreceipt_openset_train.pipeline = _base_.train_pipeline
wildreceipt_openset_test = _base_.wildreceipt_openset_test
wildreceipt_openset_test.pipeline = test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=wildreceipt_openset_train)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=wildreceipt_openset_test)
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='F1Metric',
        prefix='node',
        key='labels',
        mode=['micro', 'macro'],
        num_classes=node_num_classes,
        cared_classes=[key_node_idx, value_node_idx]),
    dict(
        type='F1Metric',
        prefix='edge',
        mode='micro',
        key='edge_labels',
        cared_classes=[1],  # Collapse to binary F1 score
        num_classes=edge_num_classes)
]
test_evaluator = val_evaluator

visualizer = dict(
    type='KIELocalVisualizer', name='visualizer', is_openset=True)
