# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9)
# running settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
checkpoint_config = dict(interval=100)
