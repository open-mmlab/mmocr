# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper', dtype='float16', optimizer=dict(type='Adam', lr=3e-5, weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
val_cfg = dict(type='ValLoop', fp16=True)
test_cfg = dict(type='TestLoop', fp16=True)
# learning rate
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=3e-5,
        by_epoch=True,
        begin=0,
        end=3,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', by_epoch=True, begin=3, end=30)
]
