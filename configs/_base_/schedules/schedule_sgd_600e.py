optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.99, weight_decay=5e-4))
train_cfg = dict(by_epoch=True, max_epochs=600)
val_cfg = dict(interval=50)
test_cfg = dict()

# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[200, 400], end=600),
]
