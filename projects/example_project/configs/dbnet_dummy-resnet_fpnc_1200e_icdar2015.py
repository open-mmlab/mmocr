_base_ = [
    '../../../configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py'
]

custom_imports = dict(imports=['projects.example_project.dummy'])

_base_.model.backbone.type = 'DummyResNet'
