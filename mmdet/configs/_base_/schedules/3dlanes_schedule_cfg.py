train_cfg = dict(max_epochs=1)

optim_wrapper = dict(optimizer=dict(type='SGD', 
                                    lr=0.01, 
                                    momentum=0.9, 
                                    weight_decay=0.0001))