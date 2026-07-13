train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=60,
)

test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=4e-4,
        weight_decay=1e-4,
    )
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=60,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=60,
    )
]