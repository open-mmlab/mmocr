optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1500, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=1500),
]
