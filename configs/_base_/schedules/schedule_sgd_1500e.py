# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.90, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)
# running settings
runner = dict(type='EpochBasedRunner', max_epochs=1500)
checkpoint_config = dict(interval=100)
