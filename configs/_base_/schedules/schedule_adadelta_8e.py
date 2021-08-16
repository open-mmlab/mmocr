# Copyright (c) OpenMMLab. All rights reserved.
# optimizer
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=dict(max_norm=0.5))
# learning policy
lr_config = dict(policy='step', step=[4, 6, 7])
total_epochs = 8
