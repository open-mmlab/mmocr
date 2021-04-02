_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_models/transformer.py',
    '../../_base_/recog_datasets/toy_dataset.py'
]

# optimizer
optimizer = dict(type='Adadelta', lr=1)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 5
