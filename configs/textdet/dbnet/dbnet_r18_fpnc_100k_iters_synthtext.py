_base_ = [
    '_base_dbnet_r18_fpnc.py',
    '../../_base_/det_datasets/synthtext.py',
    '../../_base_/textdet_default_runtime.py',
    '../../_base_/schedules/schedule_sgd_100k_iters.py',
]

# dataset settings
st_det_train = _base_.st_det_train
st_det_train.pipeline = _base_.train_pipeline
st_det_test = _base_.st_det_test
st_det_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=st_det_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=st_det_test)

test_dataloader = val_dataloader
