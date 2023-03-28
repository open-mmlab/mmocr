_base_ = [
    '/Users/wangnu/Documents/GitHub/mmocr/'
    'configs/ser/_base_/datasets/xfund_zh.py'
]

train_dataset = _base_.xfund_zh_ser_train
train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
