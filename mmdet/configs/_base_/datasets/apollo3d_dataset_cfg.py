data_root = 'data/Geo3DLanes'
dataset_type = 'mmdet.Apollo3D'
classes = ['lane']
lidar_sweeps = 2
backend_args = None
img_scale = (1920, 1080)
mask_downscale = 1
iterations = 2

train_ann_file = '/data24t_1/owais.tahir/3DLanes/mmdetection/data/Apollo_Sim_3D_Lane_Release/data_splits/standard/train.json'
val_ann_file = '/data24t_1/owais.tahir/3DLanes/mmdetection/data/Apollo_Sim_3D_Lane_Release/data_splits/standard/val.json'

img_prefix = '/data24t_1/owais.tahir/3DLanes/mmdetection/data/Apollo_Sim_3D_Lane_Release'

compose_params = dict(bboxes=False, keypoints=True, masks=True)

train_al_pipeline = [
    dict(type="Compose", params=compose_params),
    dict(type="Resize", scale=img_scale),
    # dict(type="RandomCenterCrop"),
    dict(type="HorizontalFlip"),
    dict(type="Mirror"),
    # dict(type="Perspective")
]

train_pipeline = [
    dict(type="LoadImageFromFile"), 
    dict(type="mmdet.VoxelGenerator", 
         base_height=1.78,
         y_range=0.8,
         roi_x=(-20, 20),
         roi_z=(4, 125),
         grid_res=(0.2, 0.1, 0.5)),
    # dict(type="Normalize", mean=[], std=[]),
    dict(type="mmdet.LoadLaneMasks", mask_downscale=mask_downscale, iterations=iterations),
    # dict(type="DepthCompltetion") #
    # dict(type="Resize", scale=img_scale, keep_ratio=True)
    # dict(type="Alaug", pipeline=train_al_pipeline) #
    ]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type="mmdet.LoadLaneMasks", mask_downscale=mask_downscale, iterations=iterations),
    # dict(type="Resize", scale=img_scale, keep_ratio=True)
    ] # dummy transform for now

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        img_prefix=img_prefix,
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
#         img_prefix=img_prefix,
#         metainfo=dict(classes=classes, lidar_sweeps=lidar_sweeps),
#         pipeline=val_pipeline,
#         backend_args=backend_args,
#         test_mode=True,
#     )
# )

val_dataloader = None
val_cfg = None
val_evaluator = None