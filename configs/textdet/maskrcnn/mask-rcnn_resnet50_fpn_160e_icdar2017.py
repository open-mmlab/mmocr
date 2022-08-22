_base_ = [
    'mask-rcnn_resnet50_fpn_160e_icdar2015.py',
    '../../_base_/det_datasets/icdar2017.py',
]

ic17_det_train = _base_.ic17_det_train
ic17_det_test = _base_.ic17_det_test
# use the same pipeline as icdar2015
ic17_det_train.pipeline = _base_.train_pipeline
ic17_det_test.pipeline = _base_.test_pipeline

train_dataloader = dict(dataset=ic17_det_train)
val_dataloader = dict(dataset=ic17_det_test)
test_dataloader = val_dataloader
