# optimizer
optimizer = dict(type='Adam', lr=4e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
