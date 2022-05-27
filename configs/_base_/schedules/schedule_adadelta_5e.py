optimizer = dict(type='Adadelta', lr=1.0)

train_cfg = dict(by_epoch=True, max_epochs=5)
val_cfg = dict(interval=1)
test_cfg = dict()

# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0),
]
