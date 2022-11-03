_base_ = [
    'drrg_resnet50_fpn-unet_1200e_ctw1500.py',
]

load_from = None

_base_.model.backbone = dict(
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/mnt/petrelfs/openaide-tech/gaotong/mmocr_github/r50_oclip.pth'),
)

param_scheduler = [
    dict(type='LinearLR', end=100, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=100, end=1200),
]
