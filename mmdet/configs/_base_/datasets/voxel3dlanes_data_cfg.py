data_root = 'data/Apollo_Sim_3D_Lane_Release'
dataset_type = 'mmdet.Apollo3D'


backend_args = None
img_scale = (1920, 1080)
feat_downscale = 4
iterations = 0
base_height=1.786
y_range= 7
roi_x= (-10, 10)
roi_z=(4, 80)
grid_res=(0.3, 0.3, 0.5)

train_ann_file = '/data24t_1/owais.tahir/3DLanes/mmdetection/data/Apollo_Sim_3D_Lane_Release/data_splits/lanes_in_cam/train.json'
val_ann_file = '/data24t_1/owais.tahir/3DLanes/mmdetection/data/Apollo_Sim_3D_Lane_Release/data_splits/lanes_in_cam/val.json'

img_prefix = '/data24t_1/owais.tahir/3DLanes/mmdetection/data/Apollo_Sim_3D_Lane_Release/images/'

compose_params = dict(bboxes=False, keypoints=True, masks=True)

classes = ['SingleSolid', 'SingleDash', 'DoubleSolid', 'DoubleDash', 
           'LeftDashRightSolid', 'LeftSolidRightDash', 'Curb', 'Imaginary', 'Other']
id2class = {'SingleSolid': 0,
            'SingleDash': 1,
            'DoubleSolid': 2,
            'DoubleDash': 3,
            'LeftDashRightSolid': 4,
            'LeftSolidRightDash': 5,
            'Curb': 6,
            'Imaginary': 7,
            'Other': 8}

train_pipeline = [
    dict(type="LoadImageFromFile"), 
    dict(type="mmdet.VoxelGenerator", 
            base_height=base_height,
            y_range=y_range,
            roi_x=roi_x,
            roi_z=roi_z,
            grid_res=grid_res
            ),
    dict(type="mmdet.Resize", scale=img_scale),
    dict(type="mmdet.Normalize", 
         mean=[0.56911952, 0.54184569, 0.4889298], 
         std=[0.16311612, 0.16758122, 0.1713779]
         ),
    dict(type="mmdet.LoadLaneMasks",
            iterations=iterations,
            ),
    dict(type="mmdet.Pack3DLanesInputs"),
    ]

val_pipeline = [
    dict(type="LoadImageFromFile"), 
    dict(type="mmdet.VoxelGenerator", 
            base_height=base_height,
            y_range=y_range,
            roi_x=roi_x,
            roi_z=roi_z,
            grid_res=grid_res
            ),
    dict(type="mmdet.Resize", scale=img_scale),
    dict(type="mmdet.Normalize", 
         mean=[0.56911952, 0.54184569, 0.4889298], 
         std=[0.16311612, 0.16758122, 0.1713779]
         ),
    dict(type="mmdet.LoadLaneMasks",
            iterations=iterations,
            ),
    dict(type="mmdet.Pack3DLanesInputs"),
    ] # dummy transform for now

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        img_prefix=img_prefix,
        id2class=id2class,
        feat_downscale=feat_downscale,
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
        backend_args=backend_args    
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        img_prefix=img_prefix,
        id2class=id2class,
        feat_downscale=feat_downscale,
        metainfo=dict(classes=classes),
        pipeline=val_pipeline,
        backend_args=backend_args 
    )
)

# test_evaluator = dict(
#     type='mmdet.LaneEval', 
#     mode='2D',                                            # compute '2D' or '3D' lanes metrics. only '2D' works for now
#     dist_thresh=0.5,                                      # dist threshold for matching pred lane to gt lane
#     format_only=False,                                    # just save results, don't evaluate. if True, outfile_prefix must not be None
#     outfile_prefix='../mmdet/work_dirs_test/eval_outs/',  # dir to save the gts and preds txts
#     collect_device='cpu',                                 # might break if not 'cpu'
#     prefix=None                                           # prefix for metric name
# )