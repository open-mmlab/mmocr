# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.99, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[200, 400])
# running settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
checkpoint_config = dict(interval=100)
