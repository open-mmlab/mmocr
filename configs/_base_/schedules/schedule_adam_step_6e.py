# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
# running settings
runner = dict(type='EpochBasedRunner', max_epochs=6)
checkpoint_config = dict(interval=1)
