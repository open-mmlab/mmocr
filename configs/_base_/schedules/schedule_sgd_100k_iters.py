# optimizer
optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=False)
# running settings
runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(interval=10000)
