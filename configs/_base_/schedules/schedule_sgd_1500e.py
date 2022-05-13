train_cfg = dict(by_epoch=True, max_epochs=1500)
val_cfg = dict(interval=20)  # Never evaluate
test_cfg = dict()

# learning policy
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=1500),
]
