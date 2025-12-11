train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=60,
    # val_interval=1
)

optim_wrapper = dict(optimizer=dict(type='AdamW', 
                                    lr=8e-4, 
                                    weight_decay=1e-4))