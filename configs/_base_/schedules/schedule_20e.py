# Copyright (c) OpenMMLab. All rights reserved.
_base_ = './schedule_1x.py'
# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
