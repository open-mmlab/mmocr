# BasicBlock has a little difference from official PANet
# BasicBlock in mmdet lacks RELU in the last convolution.
model = dict(
    type='PANet',
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        stem_channels=128,
        deep_stem=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        style='pytorch'),
    neck=dict(type='FPEM_FFM', in_channels=[64, 128, 256, 512]),
    det_head=dict(
        type='PANHead',
        in_channels=[128, 128, 128, 128],
        hidden_dim=128,
        out_channel=6,
        module_loss=dict(
            type='PANModuleLoss',
            loss_text=dict(type='MaskedSquareDiceLoss'),
            loss_kernel=dict(type='MaskedSquareDiceLoss'),
        ),
        postprocessor=dict(type='PANPostprocessor', text_repr_type='quad')))
