from mmengine.config import read_base

with read_base():
    from .._base_.datasets.voxel3dlanes_data_cfg import *
    from .._base_.models.voxel3dlanes_model_cfg import *
    from .._base_.schedules.voxel3dlanes_schedule_cfg import *
    from .._base_.voxel3dlanes_default_runtime import *

backend_args = None

work_dir = '../mmdet/work_dir_temp/'                    # where to store training logs and data
save_pred_pth = '../mmdet/work_dirs_test_temp'          # where to store evaluation results; this functionality is not impelemented yet, will be added soon
load_from = './mmdet/work_dir/epoch_60_adamw_ori.pth'   # pretrained checkpoint path

base_height=1.786
y_range= 7
roi_x= (-10, 10)
roi_z=(4, 80)
grid_res=(0.3, 0.3, 0.5)

feat_channel = 64
num_grids_y = int((y_range * 2) / grid_res[1])

cla_res = 25 # in cms
num_classes = int(2 * y_range*100 / cla_res)
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
    type='mmdet.Voxel3DLanes',
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
        num_classes=num_classes,
        channel_reshaped=channel_reshaped,
        inplanes=inplanes,
        eff_cla=efficientnet_cla
    ),
    loss_func = dict(
        type='mmdet.EleLoss',
        ele_range=y_range,
        voxel_ele_res=grid_res[1],
        cla_res=cla_res
    ),
    bin_loss = dict(
        type='mmdet.FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.75,            # higher alpha for sparse lanes
        reduction='mean',
        loss_weight=2.0        # slightly boost segmentation importance
    ),
    cla_res = cla_res,
    roi_x = roi_x,
    roi_z = roi_z,
    grid_res = grid_res,
    ele_range = y_range
)

test_evaluator = dict(
    type='mmdet.LaneEval3D', 
    mode='3D',                                            # compute '2D' or '3D' lanes metrics. only '2D' works for now
    dist_thresh=0.5,                                      # dist threshold for matching pred lane to gt lane
    format_only=False,                                    # just save results, don't evaluate. if True, outfile_prefix must not be None
    outfile_prefix='../mmdet/work_dirs_test/eval_outs/',  # dir to save the gts and preds txts
    collect_device='cpu',                                 # might break if not 'cpu'
    prefix=None                                           # prefix for metric name
)