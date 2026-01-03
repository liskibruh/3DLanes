from mmengine.config import read_base

with read_base():
    from .._base_.datasets.apollo3d_dataset_cfg import *
    from .._base_.models.apollo3d_model_cfg import *
    from .._base_.schedules.apollo3d_schedule_cfg import *
    from .._base_.apollo3d_default_runtime import *

# _base_ = ['../_base_/datasets/apollo3d_dataset_cfg.py',
#           '../_base_/models/3dlanes_model_cfg.py',
#           '../_base_/schedules/3dlanes_schedule_cfg.py']

backend_args = None
work_dir = '../mmdet/work_dir/'
save_pred_pth = '../mmdet/work_dirs_test'
save_stat_pth = '../mmdet/work_dirs_test/per_epoch_stats'
save_vis_pth = '../mmdet/work_dirs_test/per_epoch_ele_vis'
save_bin_pth = '../mmdet/work_dirs_test/per_epoch_bin_vis'
load_from = '/data24t_1/owais.tahir/3DLanes/mmdetection/mmdet/work_dir/epoch_35.pth'

base_height=1.786
y_range= 7 #10
roi_x= (-10, 10)
roi_z=(4, 80) #(4, 125)
grid_res=(0.3, 0.3, 0.5)

feat_channel = 64
# roi_x = torch.tensor(roi_x, dtype=torch.float32)
# roi_z = torch.tensor(roi_z, dtype=torch.float32)
# grid_res = torch.tensor(grid_res, dtype=torch.float32)
# num_grids_x = int((roi_x[1] - roi_x[0]) / grid_res[0])
# num_grids_z = int((roi_z[1] - roi_z[0]) / grid_res[2])
num_grids_y = int((y_range * 2) / grid_res[1])

# num_grids = [num_grids_x, num_grids_y, num_grids_z]
cla_res = 25 # in cms
num_classes = int(2 * y_range*100 / cla_res)
# width_mult=3.243 #1.8
channel_reshaped = feat_channel * num_grids_y
inplanes = int(channel_reshaped/8)

efficientnet_cla = dict(
    type='mmdet.EfficientNetClassification',
    stereo=False,
    width_mult=1.0,
    depth_mult=1.0,
    inplanes=inplanes,
    num_classes=num_classes,
    norm_layer='GN', # not using BN because batch_size=1
    norm_groups=4,
    act_SiLU=True   # allow negative
)

efficientnet_bin = dict(
    type='mmdet.EfficientNetClassification',
    stereo=False,
    width_mult=1.0, 
    depth_mult=1.0,
    inplanes=inplanes,
    norm_layer='GN',
    norm_groups=4,
    num_classes=1,
    act_SiLU=False,
)

model = dict(
    type='mmdet._3DLanes',
    backbone=dict(
        type='mmdet.EfficientNetFeatureBackbone',
        stereo=False,
        width_mult=1.8,
        depth_mult=2.6,
        norm_layer='GN',
        norm_groups=4
    ),
    bin_head = dict(
        type='mmdet.SegHead',
        channel_reshaped=channel_reshaped,
        inplanes=inplanes,
        roi_x=roi_x,
        roi_z=roi_z,
        grid_res=grid_res,
        y_range=y_range,
        num_classes=1,
        eff_bin=efficientnet_bin
    ),
    ele_head = dict(
        type='mmdet.EleHead',
        feat_channel=feat_channel,
        roi_x=roi_x,
        roi_z=roi_z,
        grid_res=grid_res,
        y_range=y_range,
        # cla_res=cla_res,
        num_classes=num_classes,
        channel_reshaped=channel_reshaped,
        inplanes=inplanes,
        eff_cla=efficientnet_cla
    ),
    loss_func = dict(
        type='mmdet.MyLoss',
        ele_range=y_range,
        voxel_ele_res=grid_res[1],
        cla_res=cla_res
    ),
    # bin_loss = dict(
    #     type='mmdet.CrossEntropyLoss',
    #     use_sigmoid=True,  # BCEWithLogits
    #     use_mask=False,
    #     reduction='mean',
    #     # class_weight=[1.0, 10.0], # high weight for lane pixels, addressing class imbalance (background > foreground)
    #     loss_weight=1.0,
    #     ignore_index=None,
    #     avg_non_ignore=False
    # ),
    bin_loss = dict(
        type='mmdet.FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.75,            # Recommended: higher alpha for sparse lanes
        reduction='mean',
        loss_weight=2.0        # Slightly boost segmentation importance
    ),
    cla_res = cla_res,
    roi_x = roi_x,
    roi_z = roi_z,
    grid_res = grid_res,
    ele_range = y_range
)

custom_hooks = [
    dict(
        type='mmdet.SaveVisAndStats',
        stat_out_dir=save_stat_pth,
        vis_out_dir=save_vis_pth,
        bin_out_dir=save_bin_pth,
        num_samples=1,
        roi_x=roi_x,
        roi_z=roi_z
    )
]