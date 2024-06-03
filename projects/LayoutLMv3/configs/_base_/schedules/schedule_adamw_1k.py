# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=7e-5, weight_decay=0.01))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=1000, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='OneCycleLR', eta_max=7e-5, by_epoch=False, total_steps=1000),
]
