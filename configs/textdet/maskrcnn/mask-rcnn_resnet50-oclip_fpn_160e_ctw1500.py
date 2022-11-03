_base_ = [
    'mask-rcnn_resnet50_fpn_160e_ctw1500.py',
]

load_from = None

_base_.model.cfg.backbone = dict(
    _scope_='mmocr',
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))

_base_.optim_wrapper.optimizer.lr = 0.02
