train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=60,
    # val_interval=1
)
# val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
test_evaluator = dict(type='mmdet.SaveElePredMetric', 
                      save_dir='../mmdet/work_dirs_test/3dlanes_preds',
                      vis_dir='../mmdet/work_dirs_test/vis_out/')

optim_wrapper = dict(optimizer=dict(type='AdamW', 
                                    lr=4e-4, 
                                    weight_decay=1e-4))