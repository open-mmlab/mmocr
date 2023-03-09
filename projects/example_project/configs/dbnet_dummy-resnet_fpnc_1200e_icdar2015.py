_base_ = ['mmocr::textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py']

custom_imports = dict(imports=['dummy'])

_base_.model.backbone.type = 'DummyResNet'
