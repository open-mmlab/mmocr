# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=1e-3))
train_cfg = dict(by_epoch=True, max_epochs=600)
val_cfg = dict(interval=20)
test_cfg = dict()

# learning rate
param_scheduler = [
    dict(type='PolyLR', power=0.9, end=600),
]
