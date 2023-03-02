_base_ = 'default_runtime.py'

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1000),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=1),
)

# Evaluation
val_evaluator = None
test_evaluator = None
