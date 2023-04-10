_base_ = ['./dbnet_r18_fpnc_1200e_icdar2015.py']

fp16 = dict(loss_scale='dynamic')

# learning policy
# In order to avoid non-convergence in the early stage of
# mixed-precision training, the warmup in the lr_config is set to linear,
# warmup_iters increases and warmup_ratio decreases.
lr_config = dict(warmup='linear', warmup_iters=1000, warmup_ratio=1.0 / 10)
