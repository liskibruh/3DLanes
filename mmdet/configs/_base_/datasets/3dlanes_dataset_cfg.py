data_root = 'data/Geo3DLanes'
dataset_type = 'Geo3DLanesDataset'
classes = ['lane']
lidar_sweeps = 2
backend_args = None
img_scale = (1920, 1080)

train_ann_file = '/data24t_1/owais.tahir/3DLanes/mmdetection/data/Geo3DLanes/geo3dlanes_train.pkl'
val_ann_file = '/data24t_1/owais.tahir/3DLanes/mmdetection/data/Geo3DLanes/geo3dlanes_train.pkl'

train_al_pipeline = [
    dict(type="Alaug")
]

train_pipeline = [
    dict(type="LoadImageFromFile"), 
    # dict(type="LoadLaneAnnotations"),
    # dict(type="Normalize", mean=[], std=[]),
    dict(type="Resize", scale=img_scale)
    ]
val_pipeline = [dict(type='LoadImageFromFile'),] # dummy transform for now

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        metainfo=dict(classes=classes, lidar_sweeps=lidar_sweeps),
        pipeline=train_pipeline,
        backend_args=backend_args    
    )
)
# val_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=val_ann_file,
#         metainfo=dict(classes=classes, lidar_sweeps=lidar_sweeps),
#         pipeline=val_pipeline,
#         backend_args=backend_args,
#         test_mode=True,
#     )
# )

val_dataloader = None
val_cfg = None
val_evaluator = None