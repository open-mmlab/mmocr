default_scope = 'mmocr'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
load_from = None
resume = False

val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = val_evaluator

visualizer = dict(type='TextDetLocalVisualizer', name='visualizer')
