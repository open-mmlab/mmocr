# optimizer
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[80, 128])
# running settings
runner = dict(type='EpochBasedRunner', max_epochs=160)
checkpoint_config = dict(interval=10)
