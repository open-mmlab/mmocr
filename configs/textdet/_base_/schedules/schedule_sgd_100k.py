# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000)
test_cfg = None
val_cfg = None
# learning policy
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, by_epoch=False, end=100000),
]
