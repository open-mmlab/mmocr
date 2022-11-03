_base_ = [
    'fcenet_resnet50_fpn_1500e_icdar2015.py',
]
load_from = None

_base_.model.backbone = dict(
    type='CLIPResNet',
    out_indices=(1, 2, 3),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        'https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth'  # noqa
    ),
)

_base_.train_dataloader.batch_size = 16
_base_.train_dataloader.num_workers = 24
_base_.optim_wrapper.optimizer.lr = 0.0005
