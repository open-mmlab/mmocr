# optimizer
optimizer = dict(type='Adam', lr=1e-4)

train_cfg = dict(by_epoch=True, max_epochs=600)
val_cfg = dict(interval=40)
test_cfg = dict()

# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[200, 400], end=600),
]
