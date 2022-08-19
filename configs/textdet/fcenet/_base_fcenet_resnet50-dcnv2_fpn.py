_base_ = [
    '_base_fcenet_resnet50_fpn.py',
]

model = dict(
    backbone=dict(
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=2, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    det_head=dict(
        module_loss=dict(
            type='FCEModuleLoss',
            num_sample=50,
            level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0))),
        postprocessor=dict(text_repr_type='poly', alpha=1.0, beta=2.0)))

test_pipeline = [
    dict(type='Resize', scale=(1080, 736), keep_ratio=True),
]
