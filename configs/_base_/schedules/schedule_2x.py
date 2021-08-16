# Copyright (c) OpenMMLab. All rights reserved.
_base_ = './schedule_1x.py'
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
