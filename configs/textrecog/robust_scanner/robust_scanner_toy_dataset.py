_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_models/robust_scanner.py',
    '../../_base_/recog_datasets/toy_dataset.py'
]

# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 6
