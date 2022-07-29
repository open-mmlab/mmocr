_base_ = ['satrn_academic.py']

model = dict(
    backbone=dict(type='ShallowCNN', input_channels=3, hidden_dim=256),
    encoder=dict(
        type='SATRNEncoder',
        n_layers=6,
        n_head=8,
        d_k=256 // 8,
        d_v=256 // 8,
        d_model=256,
        n_position=100,
        d_inner=256 * 4,
        dropout=0.1),
    decoder=dict(
        type='NRTRDecoder',
        n_layers=6,
        d_embedding=256,
        n_head=8,
        d_model=256,
        d_inner=256 * 4,
        d_k=256 // 8,
        d_v=256 // 8))
