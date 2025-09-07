from mmcv.transforms.loading import LoadImageFromFile
from mmengine.dataset.sampler import DefaultSampler

data_root = 'data/Geo3DLanes'
dataset_type = 'Geo3DLanesDataset'
classes = ['lane',]
backend_args = None

train_ann_file = ''
val_ann_file = ''

train_pipeline = []
val_pipeline = []

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img='train/image_data'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
        backend_args=backend_args    
    )
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img='val/image_data'),
        metainfo=dict(classes=classes),
        pipeline=val_pipeline,
        backend_args=backend_args,
        test_mode=True,
    )
)

# next-> https://mmdetection.readthedocs.io/en/latest/advanced_guides/customize_dataset.html
# create train ann file in the following format
# {
#   "metainfo": {
#     "classes": ["lane"],
#     "lidar_sweeps": 5,          # number of sweeps per sample
#     "use_pose": true
#   },
#   "data_list": [
#     {
#       "sample_id": "000123",
#       "img_path": "train/images/000123.jpg",   # optional, if you use camera
#       "height": 1080,
#       "width": 1920,

#       "lidar_points": [
#         "train/lidar/000123.bin",
#         "train/lidar/000122.bin",
#         "train/lidar/000121.bin",
#         "train/lidar/000120.bin",
#         "train/lidar/000119.bin"
#       ],

#       "pose": [
#         [1, 0, 0, 1.2],
#         [0, 1, 0, -0.3],
#         [0, 0, 1, 0.8],
#         [0, 0, 0, 1]
#       ],

#       "cam_to_velo": {
#         "front": [[...], [...], [...], [...]],
#         "left": [[...], [...], [...], [...]],
#         "right": [[...], [...], [...], [...]]
#       },

#       "instances": [
#         {
#           "lane_3d": [
#             [x1, y1, z1],
#             [x2, y2, z2],
#             [x3, y3, z3],
#             ...
#           ],
#           "lane_label": 0,
#           "ignore_flag": 0
#         }
#       ]
#     }
#   ]
# }