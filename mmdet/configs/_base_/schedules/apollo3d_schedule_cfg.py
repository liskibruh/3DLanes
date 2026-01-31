train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=60,
    # val_interval=1
)
# val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(optimizer=dict(type='AdamW', 
                                    lr=4e-4, 
                                    weight_decay=1e-4))