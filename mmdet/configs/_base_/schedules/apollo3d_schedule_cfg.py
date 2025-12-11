train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,
    # val_interval=1
)

optim_wrapper = dict(optimizer=dict(type='SGD', 
                                    lr=0.01, 
                                    momentum=0.9, 
                                    weight_decay=0.0001))